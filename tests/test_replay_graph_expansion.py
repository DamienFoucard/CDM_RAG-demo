from scripts.replay_graph_expansion import build_expander, expand, words


def entity(name, description='', relationships=()):
    return {'id': f'M::{name}', 'name': name, 'model': 'M', 'source_uri': f'{name}.json',
            'description': description, 'attributes': [], 'relationships': list(relationships)}


def relation(name, *targets):
    return {'attribute': name, 'targets': list(targets), 'description': ''}


def test_two_hop_expansion_uses_resolved_relationships_and_preserves_prefix():
    catalog = {'source_commit': 'revision', 'entities': [
        entity('Loan', 'A loan'), entity('Person', 'A person'),
        entity('Holding', 'A holding', [relation('Details', 'Loan')]),
        entity('Ownership', 'Owned holding', [relation('Owner', 'Person'), relation('Asset', 'Holding')]),
        entity('Unrelated', 'Person loan ownership', [relation('Phantom', 'Missing')]),
    ]}
    graph, chunks, groups, weights = build_expander(catalog)
    original = [chunks['M::Loan:entity'], chunks['M::Person:entity']]
    result, detail = expand('person owned loan', original, graph, chunks, groups, weights,
                            retained_prefix=1, max_expanded=3)
    ids = [item['id'] for item in result]
    assert ids[0] == 'M::Loan:entity'
    assert 'M::Ownership:relationship:1' in ids
    assert 'M::Ownership:relationship:2' in ids
    assert 'M::Unrelated:relationship:1' not in ids
    assert len(ids) == len(set(ids))
    assert len(detail['expanded']) <= 3
    assert words('own owned') == {'own'}


def test_expansion_does_not_invent_seed_for_non_entity_evidence():
    catalog = {'source_commit': 'revision', 'entities': [entity('Loan')]}
    graph, chunks, groups, weights = build_expander(catalog)
    non_entity = {'id': 'trait:example', 'text': 'loan'}
    result, detail = expand('loan', [non_entity], graph, chunks, groups, weights)
    assert result == [non_entity]
    assert detail['expanded'] == []
