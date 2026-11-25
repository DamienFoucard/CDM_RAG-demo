#!/usr/bin/env python3
"""Build a separately labeled exploratory set without explicit source-entity names."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.graph import SchemaGraph
from scripts.build_relationship_questions import B, C

CASES = [
    ("In the retail banking schema, which reference identifies the person who holds a payment card?", [(B+"FICard", "Cardholder", B+"Contact")]),
    ("Which reference identifies a person as the customer in the record connecting a customer to a financial holding?", [(B+"CustomerFinancialHolding", "Customer", B+"Contact")]),
    ("In a retail banking connection between two people, which reference identifies the person with whom it is established?", [(B+"Relationship", "ContactTo", B+"Contact")]),
    ("Which link on the banking record for holdings associated with a family group points to the group itself?", [(B+"GroupFinancialHolding", "Group", B+"Group")]),
    ("Which reference connects a comment on a knowledge-base entry to the article being commented on?", [(C+"ArticleComment", "kbArticle", C+"Article")]),
    ("Which reference identifies the team in a record saying someone belongs to that team?", [(C+"TeamMembership", "Team", C+"Team")]),
    ("A link between customers has a partner. Which reference can identify that partner as a person?", [(C+"CustomerRelationship", "partner", C+"Contact")]),
    ("How does an operating unit refer to its parent unit in the hierarchy?", [(C+"BusinessUnit", "parentBusinessUnit", C+"BusinessUnit")]),
    ("Where does a work item point to the queue that contains it?", [(C+"QueueItem", "queue", C+"Queue")]),
    ("How does an individual calendar event refer to its recurring series?", [(C+"Appointment", "series", C+"RecurringAppointment")]),
    ("What reference on a knowledge-base entry points to its principal author?", [(C+"KnowledgeArticle", "primaryAuthor", C+"User")]),
    ("Which reference connects a job role in the user's hierarchy to its parent role?", [(C+"Position", "parentPosition", C+"Position")]),
    ("Trace a work item's assigned person to that person's sales region using common schema references.", [(C+"QueueItem", "worker", C+"User"), (C+"User", "territory", C+"Territory")]),
    ("Trace the record representing someone belonging to a group through that group to the organization recorded for it.", [(C+"TeamMembership", "Team", C+"Team"), (C+"Team", "organization", C+"Organization")]),
    ("Trace a categorization of a knowledge entry to the entry itself and then its principal author.", [(C+"KnowledgeArticleCategory", "knowledgeArticle", C+"KnowledgeArticle"), (C+"KnowledgeArticle", "primaryAuthor", C+"User")]),
    ("Trace from a business customer's monetary unit to the organization recorded for that unit.", [(C+"Account", "transactionCurrency", C+"Currency"), (C+"Currency", "organization", C+"Organization")]),
]


def words(value: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', value).lower())


def main() -> None:
    catalog = json.loads((ROOT / 'seed/catalog.json').read_text(encoding='utf-8'))
    graph = SchemaGraph(catalog['entities'], catalog['source_commit'])
    by_triple = {(edge['source_id'], edge['attribute'], edge['target_id']):edge for edge in graph.edges.values()}
    rows = []
    for index, (question, triples) in enumerate(CASES, 1):
        qwords = words(question)
        for source, _relation, _target in triples:
            source_words = words(source.split('::')[-1])
            if any(qwords[i:i+len(source_words)] == source_words for i in range(len(qwords)-len(source_words)+1)):
                raise ValueError(f'challenge question {index} explicitly names gold source {source}')
        edges = [by_triple[triple] for triple in triples]
        rows.append({'id':f'challenge-{index:02}', 'split':'test', 'type':'two_hop' if len(edges) == 2 else 'direct',
                     'question':question,
                     'gold_edges':[{key:edge[key] for key in ('id','source_id','attribute','target_id','source_uri')}
                                   for edge in edges]})
    dataset = {'dataset_id':'cdm-relationships-synthetic-challenge-v1', 'source_commit':catalog['source_commit'],
               'status':'exploratory synthetic challenge, authored after the development run; not independent or human-reviewed',
               'provenance':'Assistant-authored paraphrases of selected existing graph relationships; gold edges are source-derived examples, not exhaustive relevance judgments. No source-entity name occurs verbatim in its question.',
               'split_policy':'Separate post-development challenge; some labeled edges also appear in the original pilot. This is not an independent held-out population or link-prediction benchmark.',
               'cases':rows}
    output = ROOT / 'evaluation/relationship-challenge-v1.json'
    output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} exploratory challenge questions to {output}')


if __name__ == '__main__':
    main()
