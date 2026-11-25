"""Render every captured CDM catalog fact into source-addressable passages."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("entities"), list) or not payload["entities"]:
        raise ValueError("the CDM catalog contains no entities")
    ids = [entity["id"] for entity in payload["entities"]]
    if len(ids) != len(set(ids)):
        raise ValueError("the CDM catalog contains duplicate entity IDs")
    return payload


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _trait_name(application: Any) -> str:
    return application.get("name", "") if isinstance(application, dict) else str(application)


def _trait_applications(applications: Any) -> str:
    if not isinstance(applications, list):
        return ""
    values = []
    for application in applications:
        if isinstance(application, dict):
            name = _trait_name(application)
            arguments = application.get("arguments", [])
            suffix = f" with arguments {_json(arguments)}" if arguments else ""
            if name:
                values.append(name + suffix)
        elif isinstance(application, str) and application:
            values.append(application)
    return ", ".join(values)


def _make_chunk(*, chunk_id: str, source_id: str, source_type: str, kind: str,
                title: str, text: str, uri: str, source_path: str = "",
                path: str = "", entity_id: str | None = None,
                member_paths: list[str] | None = None,
                source_label: str | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "source_id": source_id,
        "source_type": source_type,
        "kind": kind,
        "title": title,
        "source_label": source_label or (entity_id if source_type == "entity" else title),
        "text": text.strip(),
        "uri": uri,
    }
    if entity_id:
        chunk["entity_id"] = entity_id
    if source_path:
        chunk["source_path"] = source_path
    if path:
        chunk["path"] = path
    if member_paths:
        chunk["member_paths"] = member_paths
    return chunk


def render_chunks(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the complete extracted catalog, not only entities, into passages."""
    chunks: list[dict[str, Any]] = []
    entities = catalog.get("entities", [])
    entity_by_id = {entity["id"]: entity for entity in entities}

    scope = catalog.get("scope", [])
    scope_text = ", ".join(scope) if isinstance(scope, list) else str(scope)
    chunks.append(_make_chunk(
        chunk_id="cdm-catalog:overview", source_id="cdm-catalog", source_type="catalog",
        kind="catalog", title="CDM catalog scope",
        text=(f"This indexed catalog is {catalog.get('dataset_id', 'a Microsoft CDM catalog')}. "
              f"Its declared model scope is {scope_text or 'not specified'}. "
              f"It contains {len(entities)} extracted entities from pinned Microsoft CDM source "
              f"commit {catalog.get('source_commit', 'unknown')}. The indexed corpus also covers "
              "the extracted entity attributes and relationships, manifest declarations and mappings, "
              "trait definitions and applications, and entity inheritance. This is the coverage of "
              "the loaded catalog, not a claim that every artifact in the full Microsoft CDM repository "
              "has been extracted."),
        uri=catalog.get("source_repository", ""), source_label="CDM catalog"))

    for entity in entities:
        eid, uri = entity["id"], entity["source_uri"]
        label = f'{entity["name"]} ({entity["model"]}; qualified ID {eid})'
        source_path = entity.get("source_path", "")
        attributes = entity.get("attributes", [])
        summary = (f"CDM entity: {label}. Source file: {source_path}. {entity.get('description', '')} "
                   f"Selected attribute names: {', '.join(a['name'] for a in attributes[:12])}.")
        chunks.append(_make_chunk(chunk_id=f"{eid}:entity", source_id=eid, source_type="entity",
                                  entity_id=eid, kind="entity", title=label, text=summary,
                                  uri=uri, source_path=source_path, path=entity.get("path", "")))
        for start in range(0, len(attributes), 6):
            group = attributes[start:start + 6]
            attribute_text = []
            for attribute in group:
                details = [f"{attribute['name']} ({attribute.get('data_type') or 'unspecified'})"]
                if attribute.get("display_name") and attribute["display_name"] != attribute["name"]:
                    details.append(f"display name {attribute['display_name']}")
                if attribute.get("description"):
                    details.append(str(attribute["description"]))
                for key, label_text in (("is_nullable", "nullable"), ("purpose", "purpose"),
                                        ("source_name", "source name"), ("source_ordering", "source ordering")):
                    value = attribute.get(key)
                    if value is not None and value != "":
                        details.append(f"{label_text}: {value}")
                attribute_text.append("; ".join(details))
            text = (f"Attributes of CDM entity {label} at {entity.get('path', '')}: "
                    + ". ".join(attribute_text))
            paths = [a["path"] for a in group if a.get("path")]
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:attributes:{start // 6 + 1}", source_id=eid,
                source_type="entity", entity_id=eid, kind="attributes", title=f"{label} attributes",
                text=text, uri=uri, source_path=source_path,
                path=f"{entity.get('path', '')}/attributes", member_paths=paths))

        for index, relation in enumerate(entity.get("relationships", []), 1):
            targets = relation.get("target_ids") or relation.get("targets") or []
            text = (f"CDM entity {label} has an explicitly referenced relationship through "
                    f"attribute {relation['attribute']} to {', '.join(targets)}. "
                    f"{relation.get('description') or ''}")
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:relationship:{index}", source_id=eid, source_type="entity",
                entity_id=eid, kind="relationship", title=f"{label}.{relation['attribute']} relationship",
                text=text, uri=relation.get("source_uri") or uri,
                source_path=relation.get("source_path") or source_path,
                path=relation.get("path", "")))

        entity_traits = _trait_applications(entity.get("applied_traits"))
        if entity_traits:
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:trait-applications:entity", source_id=eid, source_type="entity",
                entity_id=eid, kind="trait_application", title=f"Traits applied to {label}",
                text=f"CDM entity {label} exhibits these traits: {entity_traits}.",
                uri=uri, source_path=source_path, path=entity.get("path", "")))

        attribute_trait_rows = []
        for attribute in attributes:
            applied = _trait_applications(attribute.get("applied_traits"))
            data_type = _trait_applications(attribute.get("data_type_traits"))
            if applied or data_type:
                parts = []
                if applied:
                    parts.append(f"applied traits: {applied}")
                if data_type:
                    parts.append(f"data-type traits: {data_type}")
                attribute_trait_rows.append(f"{attribute['name']} ({attribute.get('data_type') or 'unspecified'}): "
                                            + "; ".join(parts))
        for start in range(0, len(attribute_trait_rows), 8):
            rows = attribute_trait_rows[start:start + 8]
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:trait-applications:attributes:{start // 8 + 1}",
                source_id=eid, source_type="entity", entity_id=eid,
                kind="trait_application", title=f"Attribute traits of {label}",
                text=f"Traits applied to attributes of CDM entity {label}: " + "; ".join(rows),
                uri=uri, source_path=source_path, path=f"{entity.get('path', '')}/attributes"))

        base = entity.get("extends_entity")
        if isinstance(base, str) and base.strip():
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:inheritance", source_id=eid, source_type="entity",
                entity_id=eid, kind="inheritance", title=f"{label} inheritance",
                text=f"CDM entity {label} extends {base.strip()}.", uri=uri,
                source_path=source_path, path=entity.get("path", "")))

        for index, relation in enumerate(entity.get("unresolved_relationships", []), 1):
            attribute = relation.get("attribute", "")
            reference = relation.get("reference", "")
            chunks.append(_make_chunk(
                chunk_id=f"{eid}:unresolved-relationship:{index}", source_id=eid,
                source_type="entity", entity_id=eid, kind="unresolved_relationship",
                title=f"Unresolved relationship reference in {label}",
                text=(f"The extracted CDM entity {label} has an unresolved relationship reference "
                      f"from attribute {attribute} to {reference}. This catalog did not resolve the "
                      "reference to a loaded entity; this does not establish that the reference is invalid."),
                uri=relation.get("source_uri") or uri,
                source_path=relation.get("source_path") or source_path,
                path=relation.get("path", "")))

    semantics = catalog.get("semantics", {})
    manifests = semantics.get("manifests", {}) if isinstance(semantics, dict) else {}
    manifest_nodes = manifests.get("nodes", []) if isinstance(manifests, dict) else []
    manifest_by_id = {item.get("id"): item for item in manifest_nodes if isinstance(item, dict)}

    for manifest in manifest_nodes:
        manifest_id = manifest.get("id", "")
        uri = manifest.get("source_uri", "")
        source_path = manifest.get("source_path", "")
        label = f'{manifest.get("name", "Manifest")} ({manifest.get("model", "CDM")})'
        imports = manifest.get("imports", [])
        import_text = ", ".join(
            f"{item.get('corpus_path', '')}"
            + (f" as {item['moniker']}" if item.get("moniker") else "")
            for item in imports if isinstance(item, dict)
        )
        manifest_traits = _trait_applications(manifest.get("applied_traits"))
        overview = (
            f"CDM manifest {label}. {manifest.get('explanation', '')} "
            f"Document version: {manifest.get('document_version', 'unspecified')}. "
            f"JSON schema semantic version: {manifest.get('json_schema_semantic_version', 'unspecified')}. "
            f"It declares {manifest.get('declared_entity_count', 0)} entities, of which "
            f"{manifest.get('loaded_entity_count', 0)} are loaded in this catalog; it contains "
            f"{manifest.get('relationship_count', 0)} relationship mappings and "
            f"{manifest.get('sub_manifest_count', 0)} child manifests. "
            f"Imports: {import_text or 'none recorded'}. "
            f"Traits: {manifest_traits or 'none recorded'}. "
            f"Source path: {source_path}. "
            + ("This child manifest is declared but not expanded in the catalog."
               if manifest.get("declaration_only") else "")
        )
        chunks.append(_make_chunk(
            chunk_id=f"{manifest_id}:overview", source_id=manifest_id,
            source_type="manifest", kind="manifest", title=f"Manifest {label}",
            text=overview, uri=uri, source_path=source_path, path=manifest_id))

        declarations = manifest.get("declarations", [])
        for start in range(0, len(declarations), 8):
            group = declarations[start:start + 8]
            rows = []
            member_paths = []
            attached_entities = []
            for declaration in group:
                if not isinstance(declaration, dict):
                    continue
                name = declaration.get("name", "")
                detail = f"{name} ({declaration.get('type', 'LocalEntity')}) at {declaration.get('path', '')}"
                if declaration.get("entity_id"):
                    detail += f"; qualified entity ID {declaration['entity_id']}"
                traits = _trait_applications(declaration.get("applied_traits"))
                if traits:
                    detail += f"; exhibits traits: {traits}"
                for key in ("data_partitions", "data_partition_patterns", "entity_schema"):
                    if declaration.get(key):
                        detail += f"; {key.replace('_', ' ')}: {_json(declaration[key])}"
                rows.append(detail)
                if declaration.get("path"):
                    member_paths.append(f"{manifest_id}/declarations/{name}")
                if declaration.get("entity_id"):
                    attached_entities.append(declaration["entity_id"])
            chunks.append(_make_chunk(
                chunk_id=f"{manifest_id}:declarations:{start // 8 + 1}", source_id=manifest_id,
                source_type="manifest", entity_id=attached_entities[0] if len(attached_entities) == 1 else None,
                kind="manifest_declarations", title=f"Entities declared by manifest {label}",
                text=f"Entities declared in CDM manifest {label}: " + "; ".join(rows),
                uri=uri, source_path=source_path, path=f"{manifest_id}/declarations",
                member_paths=member_paths))

    # Keep manifest structure searchable independently from the full relation mappings.
    manifest_edges = manifests.get("edges", []) if isinstance(manifests, dict) else []
    edges_by_manifest: dict[str, list[dict[str, Any]]] = {}
    for edge in manifest_edges:
        if isinstance(edge, dict):
            edges_by_manifest.setdefault(str(edge.get("source_id", "")), []).append(edge)
    for manifest_id, edges in sorted(edges_by_manifest.items()):
        manifest = manifest_by_id.get(manifest_id, {})
        label = manifest.get("name", manifest_id or "CDM")
        for group_index, start in enumerate(range(0, len(edges), 12), 1):
            group = edges[start:start + 12]
            rows = []
            for edge in group:
                target = manifest_by_id.get(edge.get("target_id"), {})
                entity = entity_by_id.get(edge.get("target_id"), {})
                target_label = (target.get("name") or entity.get("name") or edge.get("target_id", ""))
                rows.append(f"{label} {edge.get('type', 'links to')} {target_label}")
            chunks.append(_make_chunk(
                chunk_id=f"{manifest_id}:structure:{group_index}",
                source_id=manifest_id or "cdm-manifests", source_type="manifest",
                kind="manifest_structure", title=f"Structure of manifest {label}",
                text=f"CDM manifest structure and declarations for {label}: " + "; ".join(rows),
                uri=manifest.get("source_uri", catalog.get("source_repository", "")),
                source_path=manifest.get("source_path", "")))

    relationships = manifests.get("relationships", []) if isinstance(manifests, dict) else []
    by_manifest_and_origin: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for relation in relationships:
        if isinstance(relation, dict):
            key = (str(relation.get("manifest_id", "")), str(relation.get("from_entity_ref", "")))
            by_manifest_and_origin.setdefault(key, []).append(relation)
    for (manifest_id, origin), rows in sorted(by_manifest_and_origin.items()):
        manifest = manifest_by_id.get(manifest_id, {})
        label = f'{manifest.get("name", "CDM")} ({manifest.get("model", "manifest")})'
        for group_index, start in enumerate(range(0, len(rows), 8), 1):
            group = rows[start:start + 8]
            lines = []
            member_paths = []
            for relation in group:
                source_attribute = relation.get("source_attribute") or "unspecified attribute"
                target_attribute = relation.get("target_attribute") or "unspecified attribute"
                target_entity = entity_by_id.get(relation.get("target_id"), {})
                target_name = target_entity.get("name") or relation.get("to_entity_ref", "")
                target_model = target_entity.get("model")
                if target_model:
                    target_name = f"{target_name} ({target_model})"
                lines.append(
                    f"{origin}.{source_attribute} maps to {relation.get('to_entity_ref', '')} "
                    f"attribute {target_attribute} (target {target_name})."
                )
                member_paths.append(f"{manifest_id}/relationships/{relation.get('id', '')}")
            origin_name = origin.rsplit("/", 1)[-1]
            origin_file = origin.split("/", 1)[0]
            origin_entity = next((item for item in entity_by_id.values()
                                  if item.get("name") == origin_name
                                  and item.get("source_path", "").rsplit("/", 1)[-1] == origin_file), None)
            chunks.append(_make_chunk(
                chunk_id=(f"{manifest_id}:relationship-mappings:{group_index}:"
                          f"{hashlib.sha256(origin.encode('utf-8')).hexdigest()[:12]}"),
                source_id=manifest_id or "cdm-manifests", source_type="manifest",
                entity_id=origin_entity.get("id") if origin_entity else None,
                kind="manifest_relationship", title=f"Relationship mappings in manifest {label}",
                text=f"The CDM manifest {label} records these relationship mappings from {origin}: "
                     + " ".join(lines), uri=manifest.get("source_uri", ""),
                source_path=manifest.get("source_path", ""),
                path=f"{manifest_id}/relationships", member_paths=member_paths))

    traits = semantics.get("traits", []) if isinstance(semantics, dict) else []
    for item in traits:
        if not isinstance(item, dict):
            continue
        definition = item.get("definition", {})
        if not isinstance(definition, dict):
            definition = {"name": str(definition)}
        name = definition.get("name") or item.get("name", "")
        parameters = definition.get("parameters", [])
        source_path = definition.get("source_path", "")
        uri = definition.get("source_uri") or catalog.get("source_repository", "")
        parent = definition.get("extends_trait")
        if isinstance(parent, dict):
            parent = parent.get("traitReference") or parent.get("namedReference") or parent.get("traitName")
        explanation = definition.get("explanation", "")
        details = (
            f"CDM trait {name}. {explanation} "
            f"Extends trait: {parent or 'none recorded'}. "
            f"Parameters: {_json(parameters)}. "
            f"Associated properties: {_json(definition.get('associated_properties', []))}. "
            f"Elevated: {definition.get('elevated')}. "
            f"Modifies attributes: {definition.get('modifies_attributes')}. "
            f"Applications in the loaded catalog: {item.get('entity_applications', 0)} entity, "
            f"{item.get('attribute_applications', 0)} attribute, "
            f"{item.get('data_type_applications', 0)} data-type, and "
            f"{item.get('manifest_applications', 0)} manifest applications."
        )
        chunks.append(_make_chunk(
            chunk_id=f"trait:{name}:definition", source_id=f"trait:{name}",
            source_type="trait", kind="trait_definition", title=f"CDM trait {name}",
            text=details, uri=uri, source_path=source_path, path=f"/traits/{name}"))

    inheritance = semantics.get("inheritance", {}) if isinstance(semantics, dict) else {}
    bases = inheritance.get("bases", []) if isinstance(inheritance, dict) else []
    for base in bases:
        if isinstance(base, str) and base:
            declaring_entity = next((entity for entity in entities
                                     if entity.get("extends_entity") == base), {})
            chunks.append(_make_chunk(
                chunk_id=f"inheritance-base:{base}", source_id=f"inheritance-base:{base}",
                source_type="inheritance", kind="inheritance_base", title=f"CDM base type {base}",
                text=f"{base} is a base entity type referenced by inheritance in the loaded CDM catalog.",
                uri=declaring_entity.get("source_uri", catalog.get("source_repository", "")),
                path=f"/inheritance/{base}"))

    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("the CDM catalog produced duplicate passage IDs")
    return chunks
