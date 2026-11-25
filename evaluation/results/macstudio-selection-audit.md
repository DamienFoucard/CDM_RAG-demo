# Offline evidence-selection audit — 24 November 2026

> The historical raw run artifacts referenced below are not included in this distribution; these results cannot be fully reproduced from this report alone.

Allowing up to two different passage kinds per source increased the number of labeled passage hits in the saved six-passage contexts from **5 to 7**, but reduced expected-entity hits from **28 to 23** across the same 12 question/arm pairs. This is not evidence that answer quality improved: the change trades source diversity for a second passage from an already represented source.

This is an offline replay of the saved raw and rewritten candidate rankings in the Mac Studio paired-answer runs, not a new retrieval or generation run. Each arm has up to 30 candidates. The existing policy keeps the first passage for each source; the alternative scans the same ranking and keeps at most two passages of different `kind` for each source. Both stop at six passages. The grouping key is `entity_id`, then `source_id`, then passage `id`, matching the existing paired-answer selector. The reported aggregate is a sum of hits, not an average of per-question recall.

| Case | Arm | Labeled passage hits, old → alternative | Expected entity hits, old → alternative |
| --- | --- | --- | --- |
| Development: group holdings | Raw | 0 → 0 | 3 → 2 |
| Development: group holdings | Rewrite | 0 → 0 | 2 → 2 |
| Development: card and customer | Raw | 0 → 1 | 2 → 1 |
| Development: card and customer | Rewrite | 1 → 1 | 2 → 1 |
| Development: holding details | Raw | 0 → 0 | 3 → 3 |
| Development: holding details | Rewrite | 0 → 0 | 3 → 3 |
| Development: common Account and Contact | Raw | 0 → 0 | 2 → 2 |
| Development: common Account and Contact | Rewrite | 2 → 2 | 2 → 2 |
| Test: customer and partner | Raw | 0 → 1 | 3 → 2 |
| Test: customer and partner | Rewrite | 1 → 1 | 3 → 2 |
| Test: card holding roles | Raw | 0 → 0 | 2 → 2 |
| Test: card holding roles | Rewrite | 1 → 1 | 1 → 1 |

Two arms gained a labeled passage, ten tied and none lost by *count*. That count conceals a material exchange: in the rewritten card-and-customer case, the alternative loses the labeled `FICard.Cardholder` passage while gaining `CustomerFinancialHolding.FinancialHolding`. Five arms lose an expected entity. Additional slots often go to trait or manifest passages; the rule cannot recover a needed passage absent from the candidate list. Two relationship passages from the same source also share a kind, so this particular policy cannot select both. The provisional labels are incomplete, so an unlabeled passage can still be useful. The six source-derived cases are small, published and not independently reviewed. No answer model or judge was rerun with these contexts.

The inputs are the development artifact, test artifact, and [candidate dataset](../paired-answer-challenge-candidate-v1.json). The artifacts are local run files under ignored `data/`; preserve them separately if the report is moved. Their SHA-256 hashes, respectively, are `944fe5dc7698eb8f7b7ce85197056b6581ece1d5ed602be5503b76eb3a00eb50`, `224fcb7198fad3eed8b65b3a61086c5ef9568bac67eb86faf963a1a1346286d6`, and `dc9beed23f8b531b221ab9708edf1e9e1363adbdd4e767a32276ed5de61e22d0`.

From the repository root, the following read-only command reproduces the per-arm counts:

```sh
python3 - <<'PY'
import json
from pathlib import Path

root = Path('.')
cases = {q['id']: q for q in json.loads(
    (root / 'evaluation/paired-answer-challenge-candidate-v1.json').read_text()
)['questions']}
for split in ('dev', 'test'):
    run = json.loads((root / f'data/macstudio-runs/paired-answer-{split}.json').read_text())
    for row in run['rows']:
        case = cases[row['id']]
        gold = set(case['evidence_passage_ids'])
        entities = set(case['expected_entity_ids'])
        for arm, result in row['arms'].items():
            old_ids = {p['passage_id'] for p in result['contexts']}
            old_entities = {p.get('entity_id') for p in result['contexts']}
            chosen, kinds_by_source = [], {}
            for candidate in result['candidates']:
                source = candidate.get('entity_id') or candidate.get('source_id') or candidate['id']
                kinds = kinds_by_source.setdefault(source, set())
                kind = candidate['kind']
                if len(kinds) >= 2 or kind in kinds:
                    continue
                kinds.add(kind)
                chosen.append(candidate)
                if len(chosen) == 6:
                    break
            new_ids = {p['id'] for p in chosen}
            new_entities = {p.get('entity_id') for p in chosen}
            print(row['id'], arm,
                  'passages', len(old_ids & gold), '->', len(new_ids & gold),
                  'entities', len(old_entities & entities), '->', len(new_entities & entities))
PY
```
