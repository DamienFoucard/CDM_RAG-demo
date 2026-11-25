#!/usr/bin/env python3
"""Build the source-checked, synthetic relationship-retrieval pilot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.graph import SchemaGraph

C = "applicationCommon::"
B = "RetailBankingCoreDataModel::"

# The author wrote these questions from the source definitions; neither the
# questions nor their example labels are independently reviewed. Questions are
# not training examples for the graph model. Each triple specifies a directed,
# explicitly extracted schema reference, not an instance-level business fact.
CASES = [
    ("development", "Which field on a common Account points to its primary Contact?", [(C+"Account", "primaryContact", C+"Contact")]),
    ("development", "How can a Contact refer to its parent customer when the customer is a business Account?", [(C+"Contact", "parentCustomer", C+"Account")]),
    ("development", "Which relationship on CustomerFinancialHolding identifies an individual customer?", [(B+"CustomerFinancialHolding", "Customer", B+"Contact")]),
    ("development", "How does a FinancialHoldingInstrument reference the financial holding it belongs to?", [(B+"FinancialHoldingInstrument", "FinancialHolding", B+"FinancialHolding")]),
    ("development", "Which relationship places a BusinessUnit under a parent business unit?", [(C+"BusinessUnit", "parentBusinessUnit", C+"BusinessUnit")]),
    ("development", "Where does a TeamMembership record identify its User?", [(C+"TeamMembership", "User", C+"User")]),
    ("development", "How does an Appointment point to the recurring appointment series it belongs to?", [(C+"Appointment", "series", C+"RecurringAppointment")]),
    ("development", "Which relationship on an Article refers to an Organization?", [(C+"Article", "organization", C+"Organization")]),
    ("development", "Which FICard relationship identifies an individual cardholder?", [(B+"FICard", "Cardholder", B+"Contact")]),
    ("development", "How does a QueueItem reference its Queue?", [(C+"QueueItem", "queue", C+"Queue")]),
    ("development", "Trace a SocialActivity to its posting SocialProfile and from that profile to its Currency.", [(C+"SocialActivity", "postFromProfile", C+"SocialProfile"), (C+"SocialProfile", "transactionCurrency", C+"Currency")]),
    ("development", "Trace an SLAItem to its SLA and from that agreement to its owning Team.", [(C+"SLAItem", "SLA", C+"SLA"), (C+"SLA", "Team_owningTeam", C+"Team")]),
    ("test", "Which relationship associates a business Account with its parent Account?", [(C+"Account", "parentAccount", C+"Account")]),
    ("test", "What link identifies the preferred system User for a Contact?", [(C+"Contact", "preferredSystemUser", C+"User")]),
    ("test", "How does CustomerFinancialHolding identify the associated FinancialHolding?", [(B+"CustomerFinancialHolding", "FinancialHolding", B+"FinancialHolding")]),
    ("test", "What connects GroupFinancialHolding to the family Group?", [(B+"GroupFinancialHolding", "Group", B+"Group")]),
    ("test", "How does GroupFinancialHolding reference a FinancialHolding?", [(B+"GroupFinancialHolding", "FinancialHolding", B+"FinancialHolding")]),
    ("test", "In the banking Relationship definition, which reference identifies the Contact who owns the relationship?", [(B+"Relationship", "ContactFrom", B+"Contact")]),
    ("test", "Which Contact is the other endpoint of a banking Relationship?", [(B+"Relationship", "ContactTo", B+"Contact")]),
    ("test", "Where does a LifeEvent refer to the person associated with it?", [(B+"LifeEvent", "Contact", B+"Contact")]),
    ("test", "Which Team does a TeamMembership record refer to?", [(C+"TeamMembership", "Team", C+"Team")]),
    ("test", "How is the administrator User linked to a Team?", [(C+"Team", "administrator", C+"User")]),
    ("test", "Which BusinessUnit is associated with a Team?", [(C+"Team", "businessUnit", C+"BusinessUnit")]),
    ("test", "How does a User record refer to its parent system User?", [(C+"User", "parentSystemUser", C+"User")]),
    ("test", "Which relationship on User points to the user's Position?", [(C+"User", "position", C+"Position")]),
    ("test", "How is a Position connected to its parent Position?", [(C+"Position", "parentPosition", C+"Position")]),
    ("test", "Which Organization does a BusinessUnit reference?", [(C+"BusinessUnit", "organization", C+"Organization")]),
    ("test", "How does an ArticleComment refer to its Article?", [(C+"ArticleComment", "kbArticle", C+"Article")]),
    ("test", "Where does a KnowledgeArticle identify its primary author User?", [(C+"KnowledgeArticle", "primaryAuthor", C+"User")]),
    ("test", "Which reference links a KnowledgeArticle to its previous article content?", [(C+"KnowledgeArticle", "previousArticleContent", C+"KnowledgeArticle")]),
    ("test", "Trace a common Account's transaction Currency to the Organization that the Currency references.", [(C+"Account", "transactionCurrency", C+"Currency"), (C+"Currency", "organization", C+"Organization")]),
    ("test", "Trace a KnowledgeArticleCategory through its KnowledgeArticle to the article's primary author User.", [(C+"KnowledgeArticleCategory", "knowledgeArticle", C+"KnowledgeArticle"), (C+"KnowledgeArticle", "primaryAuthor", C+"User")]),
    ("test", "Trace a QueueItem's worker User to that User's Territory.", [(C+"QueueItem", "worker", C+"User"), (C+"User", "territory", C+"Territory")]),
    ("test", "Trace a TeamMembership through its Team to the Team's Organization.", [(C+"TeamMembership", "Team", C+"Team"), (C+"Team", "organization", C+"Organization")]),
    ("test", "Trace a Contact through its preferred system User to that User's Queue.", [(C+"Contact", "preferredSystemUser", C+"User"), (C+"User", "queue", C+"Queue")]),
    ("test", "Trace from GroupFinancialHolding to a FinancialHolding and back to CustomerFinancialHolding that references that same holding.", [(B+"GroupFinancialHolding", "FinancialHolding", B+"FinancialHolding"), (B+"CustomerFinancialHolding", "FinancialHolding", B+"FinancialHolding")]),
]


def main() -> None:
    catalog = json.loads((ROOT / "seed/catalog.json").read_text(encoding="utf-8"))
    graph = SchemaGraph(catalog["entities"], catalog["source_commit"])
    edge_by_triple = {(edge["source_id"], edge["attribute"], edge["target_id"]): edge
                      for edge in graph.edges.values()}
    rows, seen = [], {"development": set(), "test": set()}
    for split, question, triples in CASES:
        missing = [triple for triple in triples if triple not in edge_by_triple]
        if missing:
            raise ValueError(f"question references missing extracted edge: {missing}")
        seen[split].update((source, relation) for source, relation, _target in triples)
        selected = [edge_by_triple[triple] for triple in triples]
        rows.append({"id": f"{split[:3]}-{1 + sum(row['split'] == split for row in rows):02d}",
                     "split": split, "question": question,
                     "type": "two_hop" if len(triples) == 2 else "direct",
                     "gold_edges": [{key: edge[key] for key in
                                     ("id", "source_id", "attribute", "target_id", "source_uri")}
                                    for edge in selected]})
    if seen["development"] & seen["test"]:
        raise ValueError("development and test share a source/relationship pair")
    payload = {"dataset_id": "cdm-relationships-synthetic-v1", "source_commit": catalog["source_commit"],
               "status": "synthetic pilot; not independently authored or human-reviewed",
               "provenance": "Questions were authored from the pinned extracted CDM graph and each labeled directed edge was checked against that graph and its source URI. Test questions were not independently authored. Labels are examples, not exhaustive judgments; unlabeled edges may also be relevant.",
               "split_policy": "Development and test have disjoint (source entity, relationship attribute) pairs; the graph and its observed edges remain available to all retrieval methods. This is not a held-out link-prediction benchmark.",
               "cases": rows}
    output = ROOT / "evaluation/relationship-questions-v1.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} source-checked cases to {output}")


if __name__ == "__main__":
    main()
