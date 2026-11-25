#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import posixpath
import sys
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.graph import resolve_target
from app.cdm_source import (
    UPSTREAM_REPOSITORY,
    collect_source_inputs,
    copy_source_inputs,
    verify_checkout,
)


UPSTREAM = UPSTREAM_REPOSITORY
BANK_MODEL = "RetailBankingCoreDataModel"
COMMON_MODEL = "applicationCommon"
PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))[
    "tool"]["cdm-rag"]["upstream"]["commit"]


def flatten_attributes(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, list):
        for member in value:
            result.extend(flatten_attributes(member))
    elif isinstance(value, dict):
        group = value.get("attributeGroupReference")
        if isinstance(group, dict) and isinstance(group.get("members"), list):
            result.extend(flatten_attributes(group["members"]))
        elif "name" in value and ("dataType" in value or "entity" in value):
            result.append(value)
        else:
            for key in ("members", "hasAttributes"):
                if key in value:
                    result.extend(flatten_attributes(value[key]))
    return result


def data_type(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("dataTypeReference") or value.get("entityReference") or "unspecified")
    return "unspecified"


def normalize_trait_applications(value: Any) -> list[dict[str, Any]]:
    """Keep the trait name and arguments attached to an entity/attribute."""
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append({"name": item.strip(), "arguments": []})
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("traitReference") or item.get("traitName") or item.get("namedReference")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = []
        raw_arguments = item.get("arguments", [])
        if isinstance(raw_arguments, list):
            for index, argument in enumerate(raw_arguments):
                if isinstance(argument, dict):
                    arguments.append({
                        "name": argument.get("name") or f"argument {index + 1}",
                        "value": argument.get("value", argument.get("entityReference", argument)),
                    })
                else:
                    arguments.append({"name": f"argument {index + 1}", "value": argument})
        result.append({"name": name.strip(), "arguments": arguments})
    return result


def _manifest_children(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Support both spellings found in pinned Microsoft manifest snapshots."""
    value = payload.get("subManifests", payload.get("submanifests", []))
    return value if isinstance(value, list) else []


def _safe_manifest_path(source_root: Path, parent: Path, definition: str) -> Path:
    if not definition or "\\" in definition:
        raise ValueError(f"invalid submanifest definition: {definition!r}")
    candidate = (parent / definition).resolve(strict=True)
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("submanifest path escapes the source checkout") from exc
    if not candidate.is_file() or not candidate.name.endswith(".manifest.cdm.json"):
        raise ValueError(f"submanifest definition is not a manifest file: {definition}")
    return candidate


def _manifest_entity_source_path(source_root: Path, manifest_path: Path, reference: Any) -> str | None:
    """Resolve a manifest entity reference to a repository-relative JSON file."""
    if not isinstance(reference, str) or not reference.strip():
        return None
    value = reference.strip().split("#", 1)[0]
    marker = ".cdm.json"
    if marker not in value:
        return None
    file_part = value[:value.index(marker) + len(marker)]
    if value.startswith("cdm:/"):
        candidate = source_root / "schemaDocuments" / value.removeprefix("cdm:/").lstrip("/")
    elif value.startswith("/"):
        candidate = source_root / "schemaDocuments" / value.lstrip("/")
    elif "schemaDocuments/" in file_part:
        candidate = source_root / file_part[file_part.index("schemaDocuments/"):]
    else:
        candidate = manifest_path.parent / file_part
    try:
        return candidate.resolve(strict=False).relative_to(source_root).as_posix()
    except ValueError:
        return None


def read_manifest_structure(
    roots: list[tuple[Path, str]], source_root: Path, commit: str,
    loaded_entities: list[dict[str, Any]], *, max_child_depth: int = 1,
) -> tuple[dict[str, Any], list[Path]]:
    """Capture a bounded, explicit manifest tree and its loaded declarations.

    The source catalog intentionally loads only the two root manifests. Child
    manifests are shown as CDM metadata, with declarations outside that loaded
    entity scope kept as declarations rather than misrepresented as graph nodes.
    """
    entities_by_source = {
        (entity.get("source_path", ""), entity["name"]): entity["id"]
        for entity in loaded_entities
    }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    paths: list[Path] = []
    seen: set[str] = set()
    manifest_relationships: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def visit(path: Path, model: str, parent_id: str | None, depth: int) -> str:
        relative = path.resolve(strict=True).relative_to(source_root).as_posix()
        manifest_id = "manifest:" + relative
        if manifest_id in seen:
            return manifest_id
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not parse CDM manifest: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"CDM manifest is not an object: {path}")
        seen.add(manifest_id)
        paths.append(path)
        declarations = payload.get("entities", [])
        if not isinstance(declarations, list):
            declarations = []
        declaration_rows = []
        loaded_ids = []
        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            entity_name = declaration.get("entityName")
            entity_path = declaration.get("entityPath")
            row = {
                "name": entity_name if isinstance(entity_name, str) else "",
                "path": entity_path if isinstance(entity_path, str) else "",
                "type": declaration.get("type", "LocalEntity"),
            }
            # Manifest declarations can carry more than a pointer to an entity
            # definition. Retain CDM traits and partition metadata when present.
            for source_key, catalog_key in (
                ("exhibitsTraits", "applied_traits"),
                ("dataPartitions", "data_partitions"),
                ("dataPartitionPatterns", "data_partition_patterns"),
                ("entitySchema", "entity_schema"),
            ):
                if source_key in declaration:
                    row[catalog_key] = (
                        normalize_trait_applications(declaration[source_key])
                        if source_key == "exhibitsTraits" else declaration[source_key]
                    )
            if isinstance(entity_path, str) and entity_path:
                source_file = (path.parent / entity_path.partition("/")[0]).resolve()
                try:
                    source_path = source_file.relative_to(source_root).as_posix()
                except ValueError:
                    source_path = ""
                loaded_id = entities_by_source.get((source_path, entity_name))
                if loaded_id:
                    row["entity_id"] = loaded_id
                    loaded_ids.append(loaded_id)
                    edges.append({"id": f"{manifest_id}|declares|{loaded_id}",
                                  "source_id": manifest_id, "target_id": loaded_id,
                                  "type": "declares_entity"})
            declaration_rows.append(row)
        raw_relationships = payload.get("relationships", [])
        if isinstance(raw_relationships, list):
            for relationship in raw_relationships:
                if not isinstance(relationship, dict):
                    continue
                from_ref, to_ref = relationship.get("fromEntity"), relationship.get("toEntity")
                from_source = _manifest_entity_source_path(source_root, path, from_ref)
                to_source = _manifest_entity_source_path(source_root, path, to_ref)
                from_name = from_ref.rsplit("/", 1)[-1] if isinstance(from_ref, str) else ""
                to_name = to_ref.rsplit("/", 1)[-1] if isinstance(to_ref, str) else ""
                source_id = entities_by_source.get((from_source or "", from_name))
                target_id = entities_by_source.get((to_source or "", to_name))
                from_attribute = str(relationship.get("fromEntityAttribute") or "")
                to_attribute = str(relationship.get("toEntityAttribute") or "")
                from_ref = from_ref if isinstance(from_ref, str) else ""
                to_ref = to_ref if isinstance(to_ref, str) else ""
                # Group identical declarations, but keep unresolved source refs
                # too. A loaded-subset graph must not erase valid manifest data.
                key = (manifest_id, from_ref, to_ref, from_attribute, to_attribute)
                summary = manifest_relationships.setdefault(key, {
                    "manifest_id": manifest_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "from_entity_ref": from_ref,
                    "to_entity_ref": to_ref,
                    "source_attribute": from_attribute,
                    "target_attribute": to_attribute,
                    "count": 0,
                })
                summary["count"] += 1
        submanifests = _manifest_children(payload)
        source_uri = f"{UPSTREAM}/blob/{commit}/{relative}"
        name = payload.get("manifestName") or path.name.removesuffix(".manifest.cdm.json")
        name = str(name).removesuffix(" CDM manifest")
        nodes.append({
            "id": manifest_id,
            "name": name,
            "model": model,
            "source_path": relative,
            "source_uri": source_uri,
            "explanation": payload.get("explanation", ""),
            "applied_traits": normalize_trait_applications(payload.get("exhibitsTraits")),
            "imports": [
                {"corpus_path": item.get("corpusPath", ""), **(
                    {"moniker": item["moniker"]} if isinstance(item.get("moniker"), str) else {}
                )}
                for item in payload.get("imports", [])
                if isinstance(item, dict)
            ] if isinstance(payload.get("imports", []), list) else [],
            "json_schema_semantic_version": payload.get(
                "jsonSchemaSemanticVersion", payload.get("jsonSchematicVersion", "")
            ),
            "document_version": payload.get("documentVersion", ""),
            "declared_entity_count": len(declaration_rows),
            "loaded_entity_count": len(loaded_ids),
            "loaded_entity_ids": loaded_ids,
            "declarations": declaration_rows,
            "relationship_count": len(payload.get("relationships", []))
                if isinstance(payload.get("relationships", []), list) else 0,
            "sub_manifest_count": len(submanifests),
            "expanded": depth <= max_child_depth,
        })
        if parent_id:
            edges.append({"id": f"{parent_id}|contains|{manifest_id}",
                          "source_id": parent_id, "target_id": manifest_id,
                          "type": "sub_manifest"})
        if depth < max_child_depth:
            for child in submanifests:
                if not isinstance(child, dict):
                    continue
                definition = child.get("definition") or child.get("manifestPath")
                if not isinstance(definition, str) or not definition.strip():
                    continue
                child_path = _safe_manifest_path(source_root, path.parent, definition)
                visit(child_path, model, manifest_id, depth + 1)
        else:
            # Preserve the declared topology even when we deliberately stop
            # reading deeper versioned manifests.
            for child in submanifests:
                if not isinstance(child, dict):
                    continue
                definition = child.get("definition") or child.get("manifestPath") or ""
                fallback_name = (Path(str(definition)).name.removesuffix(".manifest.cdm.json")
                                 if definition else "Submanifest")
                child_name = child.get("manifestName") or fallback_name
                child_id = f"manifest-declaration:{manifest_id}:{child_name}"
                declared_path = posixpath.normpath(posixpath.join(posixpath.dirname(relative), str(definition)))
                if declared_path.startswith("../") or declared_path.startswith("/"):
                    declared_path = ""
                declared_uri = f"{UPSTREAM}/blob/{commit}/{declared_path}" if declared_path else source_uri
                nodes.append({"id": child_id, "name": str(child_name), "model": model,
                              "source_path": declared_path, "source_uri": declared_uri,
                              "explanation": "Declared child manifest; its source is not expanded in this view.",
                              "declared_entity_count": 0, "loaded_entity_count": 0,
                              "loaded_entity_ids": [], "declarations": [],
                              "relationship_count": 0, "sub_manifest_count": 0,
                              "expanded": False, "declaration_only": True})
                edges.append({"id": f"{manifest_id}|contains|{child_id}",
                              "source_id": manifest_id, "target_id": child_id,
                              "type": "sub_manifest"})
        return manifest_id

    for path, model in roots:
        visit(path, model, None, 0)
    nodes.sort(key=lambda item: (item["model"], item["source_path"], item["name"].casefold()))
    edges.sort(key=lambda item: (item["type"], item["source_id"], item["target_id"]))
    relationships = []
    for key, row in sorted(manifest_relationships.items()):
        row["id"] = "manifest-relation:" + "|".join(key)
        relationships.append(row)
    return {"nodes": nodes, "edges": edges, "relationships": relationships}, paths


def sources(value: Any) -> set[str]:
    """Read explicit entity references, not arbitrary inline entityName labels."""
    found: set[str] = set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        for key in ('source', 'entityReference', 'entity'):
            if key in value:
                found.update(sources(value[key]))
        # Inline choice entities contain real references in their attributes;
        # their own entityName is an anonymous helper, not an external target.
        for key in ('hasAttributes', 'members'):
            for attribute in value.get(key, []):
                if isinstance(attribute, dict) and 'entity' in attribute:
                    found.update(sources(attribute['entity']))
    elif isinstance(value, list):
        for child in value:
            found.update(sources(child))
    return found


def read_definition(
    path: Path, source_root: Path, model: str, entity_name: str, commit: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse CDM definition {path}") from exc
    definitions = payload.get("definitions")
    if not isinstance(definitions, list):
        raise ValueError(f"CDM definition has no definitions list: {path}")
    matches = [item for item in definitions
               if isinstance(item, dict) and item.get("entityName") == entity_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {entity_name} definition in {path}; found {len(matches)}")
    definition = matches[0]
    raw_attributes = definition.get("hasAttributes", [])
    if not isinstance(raw_attributes, list):
        raise ValueError(f"hasAttributes must be a list for {model}::{entity_name}")
    flattened = flatten_attributes(raw_attributes)
    if raw_attributes and not flattened:
        raise ValueError(f"could not parse any attributes for {model}::{entity_name}")
    relative = path.resolve().relative_to(source_root).as_posix()
    source_uri = f"{UPSTREAM}/blob/{commit}/{relative}"
    attributes = []
    relationship_candidates = []
    for attribute in flattened:
        attribute_traits = normalize_trait_applications(attribute.get("appliedTraits"))
        data_type_traits = normalize_trait_applications(
            attribute.get("dataType", {}).get("appliedTraits")
            if isinstance(attribute.get("dataType"), dict) else None
        )
        row = {
            "name": attribute["name"],
            "display_name": attribute.get("displayName", attribute["name"]),
            "description": attribute.get("description", ""),
            "data_type": data_type(attribute.get("dataType")),
            "is_nullable": attribute.get("isNullable"),
            "purpose": attribute.get("purpose"),
            "source_name": attribute.get("sourceName"),
            "source_ordering": attribute.get("sourceOrdering"),
            "applied_traits": attribute_traits,
            "data_type_traits": data_type_traits,
        }
        attributes.append(row)
        relation = attribute.get("entity")
        if relation is not None:
            relationship_candidates.append((row, sources(relation)))
    return {
        "id": f"{model}::{entity_name}",
        "name": entity_name,
        "model": model,
        "description": definition.get("description", ""),
        "extends_entity": definition.get("extendsEntity"),
        "applied_traits": normalize_trait_applications(
            definition.get("exhibitsTraits", definition.get("appliedTraits"))
        ),
        "source_uri": source_uri,
        "source_path": relative,
        "attributes": attributes,
        "relationship_candidates": relationship_candidates,
    }


def read_manifest_entities(manifest: Path, source_root: Path, model: str, commit: str) -> tuple[list[dict[str, Any]], list[Path]]:
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse CDM manifest: {manifest}") from exc
    if not isinstance(manifest_data, dict):
        raise ValueError(f"{model} manifest is not an object")
    primary_paths = [manifest]
    entities = []
    manifest_entities = manifest_data.get("entities")
    if not isinstance(manifest_entities, list) or not manifest_entities:
        raise ValueError(f"{model} manifest has no entity list")
    for item in manifest_entities:
        if not isinstance(item, dict):
            raise ValueError(f"{model} manifest contains an invalid entity entry")
        entity_name = item.get("entityName")
        entity_path = item.get("entityPath")
        if not isinstance(entity_name, str) or not entity_name or not isinstance(entity_path, str) or not entity_path:
            raise ValueError(f"{model} manifest contains an entity without a name or path")
        definition_file, separator, target_name = entity_path.partition("/")
        if (not definition_file.endswith(".cdm.json") or Path(definition_file).name != definition_file
                or not separator or target_name != entity_name):
            raise ValueError(f"invalid manifest entity path for {entity_name}: {entity_path}")
        source_file = (manifest.parent / definition_file).resolve(strict=True)
        try:
            source_file.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"manifest entity path escapes the source checkout for {entity_name}") from exc
        row = read_definition(source_file, source_root, model, entity_name, commit)
        primary_paths.append(source_file)
        entities.append(row)
    return entities, primary_paths


def build(source_root: Path, commit: str) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    commit = verify_checkout(source_root, commit)
    schema = source_root / "schemaDocuments"
    bank_manifest = schema / "FinancialServices" / BANK_MODEL / f"{BANK_MODEL}.manifest.cdm.json"
    common_manifest = schema / "core" / COMMON_MODEL / f"{COMMON_MODEL}.manifest.cdm.json"
    bank_entities, bank_paths = read_manifest_entities(bank_manifest, source_root, BANK_MODEL, commit)
    common_entities, common_paths = read_manifest_entities(common_manifest, source_root, COMMON_MODEL, commit)
    entities = bank_entities + common_entities
    primary_paths = bank_paths + common_paths

    if len({entity["id"] for entity in entities}) != len(entities):
        raise ValueError("source manifest produces duplicate model-qualified entity IDs")

    by_id = {entity['id']: entity for entity in entities}
    for entity in entities:
        relationships = []
        unresolved = []
        for attribute, targets in entity.pop("relationship_candidates"):
            target_ids = sorted({target for reference in targets
                                 if (target := resolve_target(by_id, entity, reference)) is not None})
            for reference in sorted(targets):
                if resolve_target(by_id, entity, reference) is None:
                    unresolved.append({'attribute':attribute['name'],'reference':reference,
                                       'source_uri':entity['source_uri'],
                                       'reason':'unresolved_in_catalog_or_namespace'})
            if target_ids:
                relationships.append(
                    {
                        "attribute": attribute["name"],
                        "description": attribute.get("description", ""),
                        "targets": [by_id[target]['name'] for target in target_ids],
                        "target_ids": target_ids,
                        "source_uri": entity["source_uri"],
                    }
                )
        entity["relationships"] = relationships
        entity['unresolved_relationships'] = unresolved

    entities.sort(key=lambda item: (item["model"], item["name"].casefold()))
    manifest_structure, manifest_paths = read_manifest_structure(
        [(bank_manifest, BANK_MODEL), (common_manifest, COMMON_MODEL)],
        source_root, commit, entities, max_child_depth=2,
    )
    primary_paths.extend(manifest_paths)
    provenance = collect_source_inputs(source_root, primary_paths, commit)
    used_trait_names = {
        trait["name"]
        for entity in entities
        for trait in entity.get("applied_traits", [])
    }
    used_trait_names.update(
        trait["name"] for entity in entities for attribute in entity.get("attributes", [])
        for trait in [*attribute.get("applied_traits", []), *attribute.get("data_type_traits", [])]
    )
    used_trait_names.update(
        trait["name"] for manifest in manifest_structure["nodes"]
        for trait in manifest.get("applied_traits", [])
    )
    available_trait_definitions: dict[str, dict[str, Any]] = {}
    for source_entry in provenance["files"]:
        source_path = source_root / source_entry["path"]
        if not source_path.name.endswith(".cdm.json"):
            continue
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        definitions = payload.get("definitions", [])
        if not isinstance(definitions, list):
            continue
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            name = definition.get("traitName")
            if not isinstance(name, str):
                continue
            relative = source_entry["path"]
            available_trait_definitions[name] = {
                "name": name,
                "explanation": definition.get("explanation", ""),
                "extends_trait": definition.get("extendsTrait"),
                "parameters": definition.get("hasParameters", []),
                "elevated": definition.get("elevated"),
                "modifies_attributes": definition.get("modifiesAttributes"),
                "associated_properties": definition.get("associatedProperties", []),
                "source_path": relative,
                "source_uri": f"{UPSTREAM}/blob/{commit}/{relative}",
            }
    trait_names = set(used_trait_names)
    pending_trait_names = list(trait_names)
    while pending_trait_names:
        current_name = pending_trait_names.pop()
        definition = available_trait_definitions.get(current_name, {})
        parent = definition.get("extends_trait")
        if isinstance(parent, dict):
            parent = parent.get("traitReference") or parent.get("namedReference") or parent.get("traitName")
        if isinstance(parent, str) and parent.strip() and parent.strip() not in trait_names:
            parent_name = parent.strip()
            trait_names.add(parent_name)
            pending_trait_names.append(parent_name)
    trait_counts: dict[str, dict[str, Any]] = {}
    for entity in entities:
        for trait in entity.get("applied_traits", []):
            item = trait_counts.setdefault(trait["name"], {"entity_applications": 0, "attribute_applications": 0, "data_type_applications": 0, "manifest_applications": 0})
            item["entity_applications"] += 1
        for attribute in entity.get("attributes", []):
            for trait in attribute.get("applied_traits", []):
                item = trait_counts.setdefault(trait["name"], {"entity_applications": 0, "attribute_applications": 0, "data_type_applications": 0, "manifest_applications": 0})
                item["attribute_applications"] += 1
            for trait in attribute.get("data_type_traits", []):
                item = trait_counts.setdefault(trait["name"], {"entity_applications": 0, "attribute_applications": 0, "data_type_applications": 0, "manifest_applications": 0})
                item["data_type_applications"] += 1
    for manifest in manifest_structure["nodes"]:
        for trait in manifest.get("applied_traits", []):
            item = trait_counts.setdefault(trait["name"], {"entity_applications": 0, "attribute_applications": 0, "data_type_applications": 0, "manifest_applications": 0})
            item["manifest_applications"] += 1
    traits = []
    for name in sorted(trait_names, key=str.casefold):
        traits.append({"definition": available_trait_definitions.get(name, {"name": name}),
                       **trait_counts.get(name, {"entity_applications": 0, "attribute_applications": 0,
                                                 "data_type_applications": 0, "manifest_applications": 0})})
    return {
        "dataset_id": "microsoft-cdm-retail-banking-v2",
        "source_repository": UPSTREAM,
        "source_commit": commit,
        "provenance": provenance,
        "parser_revision": "cdm-semantics-v4",
        "scope": [BANK_MODEL, COMMON_MODEL],
        "entities": entities,
        "semantics": {
            "manifests": manifest_structure,
            "traits": traits,
            "inheritance": {
                "bases": sorted({entity["extends_entity"] for entity in entities
                                 if isinstance(entity.get("extends_entity"), str)}),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the licensed CDM reference catalog.")
    parser.add_argument("--source", type=Path, required=True, help="Microsoft/CDM checkout")
    parser.add_argument("--commit", default=UPSTREAM_COMMIT,
                        help="Exact checked-out upstream commit SHA (default: pinned project dependency)")
    parser.add_argument("--output", type=Path, required=True, help="Canonical catalog candidate path")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory for captured scoped source files")
    args = parser.parse_args()
    catalog = build(args.source, args.commit)
    source_root = args.source.resolve(strict=True)
    for target in (args.output.resolve(), args.raw_dir.resolve()):
        if target == source_root or source_root in target.parents:
            raise SystemExit("catalog and raw output paths must be outside the source checkout")
    copy_source_inputs(source_root, catalog["provenance"], args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"Wrote {len(catalog['entities'])} entities and "
        f"{len(catalog['provenance']['files'])} source files to {args.output} "
        f"(source hash {catalog['provenance']['source_hash']})"
    )


if __name__ == "__main__":
    main()
