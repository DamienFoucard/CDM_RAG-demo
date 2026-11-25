from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_relationships import GraphExplorer, evaluate_edges, neighborhood_suffix, source_key, train_transe
from scripts.build_relationship_challenge import words
from scripts.judge_relationships import judge_batch, ndcg_at_5
from app.schema.graph import SchemaGraph

ROOT = Path(__file__).resolve().parents[1]


def test_relationship_questions_reference_the_pinned_source_and_disjoint_splits():
    dataset = json.loads((ROOT / 'evaluation/relationship-questions-v1.json').read_text())
    catalog = json.loads((ROOT / 'seed/catalog.json').read_text())
    graph = SchemaGraph(catalog['entities'], catalog['source_commit'])
    by_triple = {source_key(edge): edge for edge in graph.edges.values()}
    seen = {'development': set(), 'test': set()}
    assert dataset['source_commit'] == catalog['source_commit']
    for case in dataset['cases']:
        assert len(case['gold_edges']) == (2 if case['type'] == 'two_hop' else 1)
        for gold in case['gold_edges']:
            edge = by_triple[source_key(gold)]
            assert gold['id'] == edge['id'] and gold['source_uri'] == edge['source_uri']
            path = gold['source_uri'].split('/blob/' + dataset['source_commit'] + '/', 1)[1]
            original = (ROOT / 'seed/source' / path).read_text().lower()
            assert gold['attribute'].lower() in original
            assert gold['target_id'].split('::')[-1].lower() in original
            seen[case['split']].add((gold['source_id'], gold['attribute']))
    assert not (seen['development'] & seen['test'])


def test_neighborhood_text_preserves_actual_edge_direction():
    incoming = {'id':'e1', 'source_id':'M::A', 'attribute':'belongsTo', 'target_id':'M::B'}
    outgoing = {'id':'e2', 'source_id':'M::B', 'attribute':'owner', 'target_id':'M::C'}
    index = SimpleNamespace(graph=SimpleNamespace(adjacency={'M::B':[incoming, outgoing]}))
    text = neighborhood_suffix(index, 'M::B')
    assert 'M::A --belongsTo--> M::B' in text
    assert 'M::B --owner--> M::C' in text


def test_exploratory_challenge_omits_the_gold_source_names():
    dataset = json.loads((ROOT / 'evaluation/relationship-challenge-v1.json').read_text())
    catalog = json.loads((ROOT / 'seed/catalog.json').read_text())
    graph = SchemaGraph(catalog['entities'], catalog['source_commit'])
    by_triple = {source_key(edge):edge for edge in graph.edges.values()}
    for case in dataset['cases']:
        query = words(case['question'])
        for edge in case['gold_edges']:
            assert by_triple[source_key(edge)]['id'] == edge['id']
            path = edge['source_uri'].split('/blob/' + dataset['source_commit'] + '/', 1)[1]
            original = (ROOT / 'seed/source' / path).read_text().lower()
            assert edge['attribute'].lower() in original
            assert edge['target_id'].split('::')[-1].lower() in original
            name = words(edge['source_id'].split('::')[-1])
            assert not any(query[i:i+len(name)] == name for i in range(len(query)-len(name)+1))


def test_relationship_metrics_require_both_edges_for_a_path():
    gold = [{'id':'a', 'source_id':'M::A'}, {'id':'b', 'source_id':'M::B'}]
    partial = evaluate_edges(gold, ['a', 'c'], 12, ['M::A'])
    assert partial['edge_recall'] == .5
    assert not partial['all_edges_found'] and partial['first_edge_mrr_at_20'] == 1
    beyond_budget = evaluate_edges(gold, ['a'] + [f'x{i}' for i in range(20)] + ['b'], 3, ['M::A', 'M::B'])
    assert beyond_budget['all_edges_found'] and not beyond_budget['all_edges_at_20']
    assert beyond_budget['edge_recall_at_20'] == .5


def test_transe_is_reproducible_and_returns_scores_only_for_observed_edges():
    pytest.importorskip('torch', reason='the offline TransE experiment has an optional PyTorch dependency')
    entities = {f'M::{name}': {'model':'M'} for name in 'ABCD'}
    edge_values = [
        {'id':'e1','source_id':'M::A','attribute':'r','target_id':'M::B'},
        {'id':'e2','source_id':'M::C','attribute':'r','target_id':'M::D'},
        {'id':'e3','source_id':'M::A','attribute':'s','target_id':'M::C'},
    ]
    graph = SimpleNamespace(entities=entities, edges={edge['id']:edge for edge in edge_values})
    first, metadata = train_transe(graph, dimensions=4, epochs=8, negatives=1, seed=9)
    second, _ = train_transe(graph, dimensions=4, epochs=8, negatives=1, seed=9)
    assert set(first) == {'e1', 'e2', 'e3'}
    assert first == pytest.approx(second)
    assert metadata['training_scope'].startswith('all observed graph edges')


def test_graph_search_ranks_only_observed_edges_reachable_within_two_hops():
    edge_values = [
        {'id':'e1','source_id':'M::Alpha','attribute':'owns','target_id':'M::Beta'},
        {'id':'e2','source_id':'M::Beta','attribute':'relates','target_id':'M::Gamma'},
        {'id':'e3','source_id':'M::Delta','attribute':'isolated','target_id':'M::Epsilon'},
    ]
    entities = {f'M::{name}': {'name':name} for name in ('Alpha','Beta','Gamma','Delta','Epsilon')}
    adjacency = {entity_id:[] for entity_id in entities}
    for edge in edge_values:
        adjacency[edge['source_id']].append(edge)
        adjacency[edge['target_id']].append(edge)
    catalog = {'entities':[{'id':edge['source_id'], 'relationships':[{'attribute':edge['attribute']}]}
                           for edge in edge_values]}
    class Collection:
        def get(self, ids, include):
            return {'ids':ids, 'embeddings':[[1.0, 0.0] for _ in ids]}
    class Models:
        def embed(self, _items, *, query):
            return [[1.0, 0.0]]
    index = SimpleNamespace(graph=SimpleNamespace(entities=entities, adjacency=adjacency,
                            edges={edge['id']:edge for edge in edge_values}),
                            catalog=catalog, collection=Collection())
    explorer = GraphExplorer(index, Models(), {'e1':-1, 'e2':-2, 'e3':-3})
    base, learned, info = explorer.rank('Trace Alpha to Beta', ['M::Alpha'])
    assert set(base) == {'e1', 'e2'} and set(learned) == {'e1', 'e2'}
    assert info['candidate_edges'] == 2


def test_blinded_judge_retries_bad_ids_and_requires_complete_grades():
    class Judge:
        answer_model = 'judge'
        calls = 0

        def chat_json(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'judgments':[{'id':'E99', 'grade':2, 'reason':'invented'}]}
            return {'judgments':[{'id':'E01', 'grade':2, 'reason':'directly relevant'}]}

    judge = Judge()
    result = judge_batch(judge, 'Which relation?', [{'label':'E01', 'source_id':'M::A'}])
    assert judge.calls == 2 and result[0]['id'] == 'E01'
    class InvalidJudge(Judge):
        def chat_json(self, *_args, **_kwargs):
            return {'judgments':[{'id':'E01', 'grade':True, 'reason':'invalid boolean'}]}
    with pytest.raises(RuntimeError, match='complete valid batch'):
        judge_batch(InvalidJudge(), 'Which relation?', [{'label':'E01'}])


def test_judge_ndcg_uses_the_pooled_ideal():
    grades = {'useful':2, 'context':1, 'irrelevant':0}
    assert ndcg_at_5(['useful', 'context', 'irrelevant'], grades) == pytest.approx(1)
    assert ndcg_at_5(['irrelevant', 'context', 'useful'], grades) < 1
    assert ndcg_at_5(['irrelevant'], {'irrelevant':0}) is None
