"""Bounded schema neighborhoods from explicit CDM references, with source lineage."""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections import deque
from typing import Any

from app.schema.catalog import canonical_path, entity_matches_address


def _source_path(uri: str) -> str:
    marker = 'schemaDocuments/'
    if marker in uri:
        return marker + uri.split(marker, 1)[1].split('#', 1)[0]
    return uri


def resolve_target(entities: dict[str, dict], source: dict, reference: str) -> str | None:
    """Bare references stay in their source model; explicit file paths can cross it."""
    if reference in entities:
        return reference
    local = f"{source['model']}::{reference}"
    if local in entities:
        return local
    if '.cdm.json' not in reference:
        return None
    file_part, _, entity_part = reference.partition('.cdm.json')
    file_part += '.cdm.json'
    if file_part.startswith('cdm:/'):
        target_path = 'schemaDocuments/' + file_part[len('cdm:/'):].lstrip('/')
    elif 'schemaDocuments/' in file_part:
        target_path = _source_path(file_part)
    else:
        target_path = posixpath.normpath(posixpath.join(
            posixpath.dirname(_source_path(source['source_uri'])), file_part
        ))
    target_name = entity_part.strip('/')
    matches = [eid for eid, entity in entities.items()
               if _source_path(entity['source_uri']) == target_path
               and (not target_name or entity['name'] == target_name)]
    return matches[0] if len(matches) == 1 else None


class SchemaGraph:
    def __init__(self, entities: list[dict], revision: str, source: str = 'cdm',
                 semantics: dict[str, Any] | None = None):
        normalized_entities = []
        for entity in entities:
            normalized = dict(entity)
            normalized.setdefault('path', canonical_path(normalized['model'], normalized['name']))
            normalized.setdefault('source_path', _source_path(normalized['source_uri']))
            for attribute in normalized.get('attributes', []):
                attribute.setdefault('path', canonical_path(
                    normalized['model'], normalized['name'], 'attributes', attribute['name']))
                attribute.setdefault('source_path', normalized['source_path'])
            for relation in normalized.get('relationships', []):
                relation.setdefault('path', canonical_path(
                    normalized['model'], normalized['name'], 'relationships', relation['attribute']))
                relation.setdefault('source_path', normalized['source_path'])
            for unresolved in normalized.get('unresolved_relationships', []):
                attribute = unresolved.get('attribute')
                unresolved.setdefault(
                    'path', canonical_path(normalized['model'], normalized['name'], 'relationships', attribute)
                    if isinstance(attribute, str) and attribute else normalized['path'])
                unresolved.setdefault('source_path', normalized['source_path'])
            normalized_entities.append(normalized)
        self.entities = {e['id']: e for e in normalized_entities}
        if len(self.entities) != len(entities):
            raise ValueError('schema entity IDs must be unique')
        self.revision = revision
        self.source = source
        self.semantics = semantics or {}
        self.projection_revision = hashlib.sha256(json.dumps(
            {"entities": normalized_entities, "semantics": self.semantics},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.edges: dict[str, dict] = {}
        self.adjacency: dict[str, list[dict]] = {eid: [] for eid in self.entities}
        self.unresolved: list[dict] = []
        for entity in normalized_entities:
            for unresolved in entity.get('unresolved_relationships', []):
                attribute = unresolved.get('attribute')
                path = (canonical_path(entity['model'], entity['name'], 'relationships', attribute)
                        if isinstance(attribute, str) and attribute else entity['path'])
                self.unresolved.append({**unresolved, 'source_id': entity['id'], 'path': path,
                                        'source_path': entity['source_path']})
            for relation in entity.get('relationships', []):
                references = relation.get('target_ids', relation.get('targets', []))
                for reference in references:
                    target = resolve_target(self.entities, entity, reference)
                    if target is None:
                        self.unresolved.append({'source_id':entity['id'], 'attribute':relation['attribute'],
                                                'reference':reference, 'reason':'unresolved_in_catalog',
                                                'path':relation.get('path'),
                                                'source_path':relation.get('source_path',entity['source_path']),
                                                'source_uri':relation.get('source_uri',entity['source_uri'])})
                        continue
                    identity = f"{entity['id']}|{relation['attribute']}|{target}"
                    edge_id = 'edge:' + hashlib.sha256(identity.encode()).hexdigest()[:20]
                    if edge_id in self.edges:
                        continue
                    edge = {'id':edge_id, 'source_id':entity['id'], 'target_id':target,
                            'attribute':relation['attribute'], 'description':relation.get('description',''),
                            'target_attribute':relation.get('target_attribute'),
                            'path':relation.get('path'),
                            'source_path':relation.get('source_path',entity.get('source_path')),
                            'target_path':self.entities[target].get('path'),
                            'source_uri':relation.get('source_uri',entity['source_uri']),
                            'source_revision':revision}
                    self.edges[edge_id] = edge
                    self.adjacency[entity['id']].append(edge)
                    if target != entity['id']:
                        self.adjacency[target].append(edge)
        for edges in self.adjacency.values():
            edges.sort(key=lambda edge:edge['id'])

    def semantic_layers(self) -> dict[str, Any]:
        """Build graph-ready, separately typed views of CDM-native semantics."""
        stored = self.semantics if isinstance(self.semantics, dict) else {}
        trait_definitions = stored.get("traits", [])
        traits = []
        for item in trait_definitions if isinstance(trait_definitions, list) else []:
            if not isinstance(item, dict):
                continue
            definition = item.get("definition", {})
            if not isinstance(definition, dict):
                definition = {"name": str(definition)}
            name = definition.get("name") or item.get("name")
            if not isinstance(name, str) or not name:
                continue
            trait = {
                **definition,
                "id": "trait:" + name,
                "name": name,
                "entity_applications": int(item.get("entity_applications", 0)),
                "attribute_applications": int(item.get("attribute_applications", 0)),
                "data_type_applications": int(item.get("data_type_applications", 0)),
                "manifest_applications": int(item.get("manifest_applications", 0)),
            }
            traits.append(trait)
        traits_by_name = {item["name"]: item for item in traits}
        trait_inheritance_edges = []
        for trait in traits:
            parent = trait.get("extends_trait")
            if isinstance(parent, dict):
                parent = parent.get("traitReference") or parent.get("namedReference") or parent.get("traitName")
            if isinstance(parent, str) and parent in traits_by_name:
                trait_inheritance_edges.append({
                    "id": f"trait-inheritance:trait:{parent}|trait:{trait['name']}",
                    "source_id": "trait:" + trait["name"],
                    "target_id": "trait:" + parent,
                    "type": "extends_trait",
                })
        trait_edges: dict[tuple[str, str], dict[str, Any]] = {}
        base_names = set()
        inheritance_edges = []
        for entity in self.entities.values():
            entity_id = entity["id"]
            for trait in entity.get("applied_traits", []):
                name = trait.get("name") if isinstance(trait, dict) else None
                if not isinstance(name, str):
                    continue
                edge = trait_edges.setdefault((entity_id, name), {
                    "id": f"trait-use:{entity_id}|{name}",
                    "source_id": entity_id,
                    "target_id": "trait:" + name,
                    "trait_name": name,
                    "entity_application_count": 0,
                    "attribute_application_count": 0,
                    "data_type_application_count": 0,
                    "attribute_names": [],
                    "data_type_attribute_names": [],
                })
                edge["entity_application_count"] += 1
            for attribute in entity.get("attributes", []):
                for trait in attribute.get("applied_traits", []):
                    name = trait.get("name") if isinstance(trait, dict) else None
                    if not isinstance(name, str):
                        continue
                    edge = trait_edges.setdefault((entity_id, name), {
                        "id": f"trait-use:{entity_id}|{name}",
                        "source_id": entity_id,
                        "target_id": "trait:" + name,
                        "trait_name": name,
                        "entity_application_count": 0,
                        "attribute_application_count": 0,
                        "data_type_application_count": 0,
                        "attribute_names": [],
                        "data_type_attribute_names": [],
                    })
                    edge["attribute_application_count"] += 1
                    edge["attribute_names"].append(attribute.get("name", ""))
                for trait in attribute.get("data_type_traits", []):
                    name = trait.get("name") if isinstance(trait, dict) else None
                    if not isinstance(name, str):
                        continue
                    edge = trait_edges.setdefault((entity_id, name), {
                        "id": f"trait-use:{entity_id}|{name}",
                        "source_id": entity_id,
                        "target_id": "trait:" + name,
                        "trait_name": name,
                        "entity_application_count": 0,
                        "attribute_application_count": 0,
                        "data_type_application_count": 0,
                        "attribute_names": [],
                        "data_type_attribute_names": [],
                    })
                    edge["data_type_application_count"] += 1
                    edge["data_type_attribute_names"].append(attribute.get("name", ""))
            base = entity.get("extends_entity")
            if isinstance(base, str) and base.strip():
                base = base.strip()
                base_id = "base:" + base
                base_names.add(base)
                inheritance_edges.append({
                    "id": f"inheritance:{entity_id}|{base_id}",
                    "source_id": entity_id,
                    "target_id": base_id,
                    "type": "extends_entity",
                })
        for edge in trait_edges.values():
            edge["attribute_names"] = sorted(set(edge["attribute_names"]), key=str.casefold)
            edge["data_type_attribute_names"] = sorted(set(edge["data_type_attribute_names"]), key=str.casefold)
        inheritance_bases = [{"id": "base:" + name, "name": name, "kind": "base_type"}
                             for name in sorted(base_names, key=str.casefold)]
        manifests = stored.get("manifests", {})
        if not isinstance(manifests, dict):
            manifests = {}
        return {
            "manifests": {
                "nodes": manifests.get("nodes", []),
                "edges": manifests.get("edges", []),
                "relationships": manifests.get("relationships", []),
            },
            "traits": {
                "nodes": sorted(traits, key=lambda item: item["name"].casefold()),
                "edges": sorted(trait_edges.values(), key=lambda item: (item["trait_name"].casefold(), item["source_id"])),
                "inheritance_edges": sorted(
                    trait_inheritance_edges, key=lambda item: (item["target_id"], item["source_id"])
                ),
            },
            "inheritance": {
                "nodes": inheritance_bases,
                "edges": sorted(inheritance_edges, key=lambda item: (item["target_id"], item["source_id"])),
            },
        }

    def restrict(self, allowed_entity_ids: set[str], allowed_edge_ids: set[str]) -> "SchemaGraph":
        """Build a request-local graph containing only authorized nodes and edges."""
        entities = []
        for entity_id, entity in self.entities.items():
            if entity_id not in allowed_entity_ids:
                continue
            visible = dict(entity)
            visible_relationships = []
            for relation in entity.get("relationships", []):
                targets = list(dict.fromkeys(
                    edge["target_id"] for edge_id, edge in self.edges.items()
                    if edge_id in allowed_edge_ids
                    and edge["source_id"] == entity_id
                    and edge["attribute"] == relation.get("attribute")
                ))
                if not targets:
                    continue
                visible_relationships.append({
                    **relation,
                    "target_ids": targets,
                    "target_paths": [self.entities[target]["path"] for target in targets],
                })
            visible["relationships"] = visible_relationships
            entities.append(visible)
        scoped = SchemaGraph(entities, self.revision, self.source)
        scoped.edges = {edge_id: edge for edge_id, edge in scoped.edges.items()
                        if edge_id in allowed_edge_ids}
        scoped.adjacency = {
            entity_id: [edge for edge in edges if edge["id"] in scoped.edges]
            for entity_id, edges in scoped.adjacency.items()
        }
        # Excluded targets can appear as unresolved references during projection.
        # They are omitted so an unauthorized endpoint cannot leak via diagnostics.
        scoped.unresolved = []
        return scoped

    def _node(self, entity_id: str) -> dict:
        entity = self.entities[entity_id]
        return {k: entity[k] for k in ('id','name','model','description','path','source_path','source_uri')} | {
            'source_revision': self.revision,
        }

    def entity_ids_for_address(self, address: str, *, prefix: bool = False) -> list[str]:
        return [entity_id for entity_id, entity in self.entities.items()
                if entity_matches_address(entity, address, prefix=prefix)]

    def neighborhood(self, seeds: list[str], depth: int = 1, max_nodes: int = 20,
                     max_edges: int = 40, max_context_bytes: int = 16000,
                     preferred_nodes: set[str] | None = None,
                     seed_origins: dict[str,str] | None = None) -> dict[str, Any]:
        if depth not in (0,1,2) or not 1 <= max_nodes <= 100 or not 1 <= max_edges <= 200:
            raise ValueError('invalid graph depth or bounds')
        if not 1024 <= max_context_bytes <= 100000:
            raise ValueError('graph context budget must be 1024..100000 bytes')
        result: dict[str, Any] = {'source':self.source, 'source_revision':self.revision,
            'graph_revision':self.projection_revision,
            'seed_origins':seed_origins or {seed:'explicit_entity_id' for seed in seeds},
            'seed_entity_ids':[], 'nodes':[], 'edges':[], 'paths':[], 'unresolved':[],
            'truncated':False, 'coverage':'loaded_catalog_subset', 'depth':depth,
            'budgets':{'nodes':max_nodes,'edges':max_edges,'context_bytes':max_context_bytes}}
        node_ids: set[str] = set()
        edge_ids: set[str] = set()

        def fits(candidate: dict) -> bool:
            return len(json.dumps(candidate,ensure_ascii=False).encode()) <= max_context_bytes

        for seed in dict.fromkeys(seeds):
            if seed not in self.entities:
                continue
            node = self._node(seed)
            trial = {**result,'seed_entity_ids':result['seed_entity_ids']+[seed],
                     'nodes':result['nodes']+[node]}
            if len(node_ids) >= max_nodes or not fits(trial):
                result['truncated'] = True
                continue
            result.update(trial)
            node_ids.add(seed)
        queue = deque((seed,[],frozenset([seed])) for seed in result['seed_entity_ids'])
        seen_paths: set[tuple] = set()
        while queue:
            current, steps, visited = queue.popleft()
            if len(steps) >= depth:
                continue
            adjacent = sorted(self.adjacency[current], key=lambda e:(
                -int(e['source_id'] in (preferred_nodes or set()))
                -int(e['target_id'] in (preferred_nodes or set())),e['id']))
            for edge in adjacent:
                target = edge['target_id'] if current == edge['source_id'] else edge['source_id']
                if target in visited and not (target == current and not steps):
                    continue
                step = {'edge_id':edge['id'],'from':current,'to':target,
                        'direction':'outgoing' if current == edge['source_id'] else 'incoming'}
                path = steps + [step]
                # For bounded one/two-hop walks the edge set identifies the
                # same evidence discovered from either end. Retain it once.
                identity = tuple(sorted(s['edge_id'] for s in path))
                if identity in seen_paths:
                    continue
                seen_paths.add(identity)
                new_nodes = [] if target in node_ids else [self._node(target)]
                new_edges = [] if edge['id'] in edge_ids else [edge]
                address_ids = [path[0]['from']] + [step['to'] for step in path]
                address_path = [self.entities[entity_id]['path'] for entity_id in address_ids]
                trial = {**result,'nodes':result['nodes']+new_nodes,'edges':result['edges']+new_edges,
                         'paths':result['paths']+[{'seed':path[0]['from'],'steps':path,
                                                   'path':address_path}]}
                if len(node_ids)+len(new_nodes)>max_nodes or len(edge_ids)+len(new_edges)>max_edges or not fits(trial):
                    result['truncated'] = True
                    continue
                result.update(trial)
                node_ids.add(target)
                edge_ids.add(edge['id'])
                pending = (target,path,visited|{target})
                # Reach explicitly named endpoints before spending the context
                # budget on unrelated fan-out from the remaining seed nodes.
                if target in (preferred_nodes or set()): queue.appendleft(pending)
                else: queue.append(pending)
        for reference in self.unresolved:
            if reference['source_id'] in node_ids:
                trial = {**result,'unresolved':result['unresolved']+[reference]}
                if fits(trial): result.update(trial)
                else: result['truncated'] = True
        return result

    def named_entities(self, query: str) -> set[str]:
        return {eid for eid,entity in self.entities.items()
                if re.search(r'\b'+re.escape(entity['name'])+r'\b',query,re.I)
                and (entity['name'].casefold() not in {'relationship','entity','attribute','type','model','schema'}
                     or re.search(r'\b'+re.escape(entity['name'])+r'\b',query))}

    def evidence(self, neighborhood: dict, limit: int = 4, query: str = '') -> list[dict]:
        """Each path is atomic; every edge keeps its original direction and source."""
        units = []
        if limit < 1: return units
        seen = set()
        named = self.named_entities(query)
        def ordering(path):
            nodes = {step[key] for step in path['steps'] for key in ('from','to')}
            return (-len(nodes & named), -len(path['steps']))
        paths = sorted(neighborhood['paths'],key=ordering)
        for path in paths:
            ids = tuple(step['edge_id'] for step in path['steps'])
            if all(step['direction']=='incoming' for step in path['steps']):
                ids = tuple(reversed(ids))
            if ids in seen: continue
            seen.add(ids)
            edges = [self.edges[eid] for eid in ids]
            node_ids = list(dict.fromkeys(eid for edge in edges for eid in (edge['source_id'],edge['target_id'])))
            addresses = [self._node(entity_id) for entity_id in node_ids]
            address_path = list(dict.fromkeys(
                self.entities[entity_id]['path'] for entity_id in
                ([path['seed']] + [step['to'] for step in path['steps']])
            ))
            sources = list(dict.fromkeys([edge['source_uri'] for edge in edges] +
                                        [self.entities[eid]['source_uri'] for eid in node_ids]))
            clauses = [f"{edge['source_id']}.{edge['attribute']} -> {edge['target_id']}"
                       f"{'.'+edge['target_attribute'] if edge['target_attribute'] else ''}. {edge['description']}"
                       for edge in edges]
            descriptions = ' ; '.join(f"{eid}: {self.entities[eid]['description']}" for eid in node_ids)
            unit = {'id':'graph:' + hashlib.sha256('|'.join(ids).encode()).hexdigest()[:20],
                'entity_id':path['seed'], 'kind':'relationship_path', 'retrieval_role':'graph_neighbor',
                'text':'Explicit schema path (edge arrows preserve source direction): ' + ' ; '.join(clauses) +
                       '. Entity definitions: ' + descriptions,
                'uri':sources[0], 'source_uris':sources, 'source':self.source,
                'source_revision':self.revision, 'path_edges':edges, 'traversal':path['steps'],
                'path':' -> '.join(address_path), 'addresses':addresses,
                'source_paths':list(dict.fromkeys(item['source_path'] for item in addresses)),
                'graph_revision':self.projection_revision,
                'score':0.0,'lexical_score':0.0,'embedding_score':0.0,
                **({'fictional':True} if self.source=='sample' else {})}
            if len(json.dumps(units+[unit],ensure_ascii=False).encode()) > neighborhood['budgets']['context_bytes']:
                neighborhood['truncated'] = True
                continue
            units.append(unit)
            if len(units) >= limit: break
        return units


def enrich(graph: SchemaGraph, results: list[dict], depth: int, query: str = '',
           max_context_bytes: int = 16000, path: str | None = None,
           path_prefix: str | None = None) -> dict:
    named = graph.named_entities(query)
    ordered_names = sorted(named,key=lambda eid:(
        re.search(r'\b'+re.escape(graph.entities[eid]['name'])+r'\b',query,re.I).start(),eid))
    addressed = graph.entity_ids_for_address(path, prefix=False) if path else []
    prefixed = graph.entity_ids_for_address(path_prefix, prefix=True) if path_prefix else []
    seeds = list(dict.fromkeys(addressed + prefixed + ordered_names +
                               [row['entity_id'] for row in results]))[:4]
    preferred = named | set(addressed) | set(prefixed)
    origins = {
        eid: ('path_match' if eid in addressed else 'path_prefix_match' if eid in prefixed
              else 'query_name_match' if eid in named else 'retrieval_match')
        for eid in seeds
    }
    neighborhood = graph.neighborhood(seeds,depth,max_context_bytes=max_context_bytes,preferred_nodes=preferred,
        seed_origins=origins)
    return {'neighborhood':neighborhood,'neighbor_evidence':graph.evidence(neighborhood,query=query) if depth else []}
