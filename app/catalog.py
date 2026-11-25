from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def canonical_path(model: str, name: str, *parts: str) -> str:
    """Return the stable logical address for a CDM entity or child item."""
    segments = (model, name, *parts)
    if any(not isinstance(segment, str) or not re.fullmatch(r"[A-Za-z0-9_.~-]+", segment)
           for segment in segments):
        raise ValueError("CDM path segments must use letters, digits, dot, dash, or underscore")
    return "/" + "/".join(segments)


def source_path(uri: str) -> str:
    """Extract the repository-relative source file path from a pinned CDM URL."""
    parsed = urlsplit(uri)
    parts = parsed.path.split("/")
    try:
        blob_index = parts.index("blob")
    except ValueError:
        return parsed.path.lstrip("/") or uri
    return "/".join(parts[blob_index + 2:])


def normalize_address(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().strip("/")
    if not raw:
        return None
    parts = raw.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9_.~-]+", part) or part in {".", ".."}
           for part in parts):
        raise ValueError("path must contain only valid CDM address segments")
    return "/" + "/".join(parts)


def path_is_at_or_below(candidate: str, anchor: str) -> bool:
    """Compare hierarchical paths on segment boundaries, not string prefixes."""
    candidate = candidate.strip("/")
    anchor = anchor.strip("/")
    return candidate == anchor or candidate.startswith(anchor + "/")


def entity_paths(entity: dict[str, Any]) -> list[str]:
    paths = [entity.get("path", ""), entity.get("source_path", "")]
    paths.extend(attribute.get("path", "") for attribute in entity.get("attributes", []))
    paths.extend(relation.get("path", "") for relation in entity.get("relationships", []))
    return [path for path in paths if isinstance(path, str) and path]


def entity_matches_address(entity: dict[str, Any], address: str, *, prefix: bool) -> bool:
    """Match an entity root, a child address, or its upstream source location."""
    address = address.strip("/")
    for candidate in entity_paths(entity):
        candidate = candidate.strip("/")
        if prefix:
            if path_is_at_or_below(candidate, address):
                return True
        elif candidate == address:
            return True
    return False


def chunk_matches_address(chunk: dict[str, Any], address: str, *, prefix: bool) -> bool:
    address = address.strip("/")
    candidates = [chunk.get("path", ""), chunk.get("source_path", "")]
    candidates.extend(chunk.get("member_paths", []))
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        candidate = candidate.strip("/")
        if prefix:
            if path_is_at_or_below(candidate, address):
                return True
        elif candidate == address or path_is_at_or_below(candidate, address):
            return True
    return False


def address_exists(entities: list[dict[str, Any]], chunks: list[dict[str, Any]],
                   address: str, *, prefix: bool = False,
                   allowed_chunk_ids: set[str] | None = None) -> bool:
    if prefix:
        if any(entity_matches_address(entity, address, prefix=True) for entity in entities):
            return True
        return any(
            (allowed_chunk_ids is None or chunk["id"] in allowed_chunk_ids)
            and chunk_matches_address(chunk, address, prefix=True)
            for chunk in chunks
        )
    target = address.strip("/")
    if any(path.strip("/") == target for entity in entities for path in entity_paths(entity)):
        return True
    for chunk in chunks:
        if allowed_chunk_ids is not None and chunk["id"] not in allowed_chunk_ids:
            continue
        candidates = [chunk.get("path", ""), chunk.get("source_path", "")]
        candidates.extend(chunk.get("member_paths", []))
        if any(isinstance(candidate, str) and candidate.strip("/") == target for candidate in candidates):
            return True
    return False


def load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload["entities"]
    for entity in entities:
        expected = canonical_path(entity["model"], entity["name"])
        if entity.get("path", expected) != expected:
            raise ValueError(f"invalid logical path for {entity['id']}")
        entity["path"] = expected
        expected_source = source_path(entity["source_uri"])
        if entity.get("source_path", expected_source) != expected_source:
            raise ValueError(f"invalid source path for {entity['id']}")
        entity["source_path"] = expected_source
        for attribute in entity.get("attributes", []):
            expected_attribute_path = canonical_path(entity["model"], entity["name"], "attributes", attribute["name"])
            if attribute.get("path", expected_attribute_path) != expected_attribute_path:
                raise ValueError(f"invalid attribute path for {entity['id']}.{attribute['name']}")
            attribute["path"] = expected_attribute_path
            attribute["source_path"] = expected_source
        for relation in entity.get("relationships", []):
            expected_relation_path = canonical_path(entity["model"], entity["name"], "relationships", relation["attribute"])
            if relation.get("path", expected_relation_path) != expected_relation_path:
                raise ValueError(f"invalid relationship path for {entity['id']}.{relation['attribute']}")
            relation["path"] = expected_relation_path
            relation["source_path"] = expected_source
    return entities


def text_chunks(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for entity in entities:
        label = f"{entity['name']} ({entity['model']})"
        summary = " ".join(
            part
            for part in (
                f"Entity {label}.",
                f"Canonical path {entity['path']}.",
                entity.get("description", ""),
                "Attributes: " + "; ".join(
                    f"{a['name']} ({a.get('data_type', 'unspecified')}): {a.get('description', '')}"
                    for a in entity.get("attributes", [])[:30]
                ),
            )
            if part
        )
        chunks.append(
            {
                "id": f"{entity['id']}:summary",
                "entity_id": entity["id"],
                "kind": "entity",
                "text": summary,
                "uri": entity["source_uri"],
                "path": entity["path"],
                "entity_path": entity["path"],
                "source_path": entity["source_path"],
            }
        )
        for index in range(0, len(entity.get("attributes", [])), 12):
            attributes = entity["attributes"][index : index + 12]
            text = f"Attributes of {label}: " + "; ".join(
                f"{a['name']} ({a.get('data_type', 'unspecified')}): {a.get('description', '')}"
                for a in attributes
            )
            attribute_paths = [a["path"] for a in attributes]
            chunks.append(
                {
                    "id": f"{entity['id']}:attributes:{index // 12 + 1}",
                    "entity_id": entity["id"],
                    "kind": "attributes",
                    "text": f"{text} Canonical paths: " + "; ".join(attribute_paths),
                    "uri": entity["source_uri"],
                    "path": canonical_path(entity["model"], entity["name"], "attributes"),
                    "entity_path": entity["path"],
                    "member_paths": attribute_paths,
                    "source_path": entity["source_path"],
                }
            )
        for index, relation in enumerate(entity.get("relationships", [])):
            text = (
                f"{label} has relationship attribute {relation['attribute']} to "
                f"{', '.join(relation['targets'])}. {relation.get('description', '')}"
            )
            chunks.append(
                {
                    "id": f"{entity['id']}:relationship:{index + 1}",
                    "entity_id": entity["id"],
                    "kind": "relationship",
                    "text": text,
                    "uri": relation.get("source_uri", entity["source_uri"]),
                    "path": relation["path"],
                    "entity_path": entity["path"],
                    "source_path": relation["source_path"],
                    "target_entity_ids": relation.get("target_ids", relation.get("targets", [])),
                    "target_paths": relation.get("target_paths", []),
                }
            )
    return chunks


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())
