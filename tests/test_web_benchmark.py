from __future__ import annotations

import pytest

from app.ollama import ModelError
from scripts.benchmark_web import summarize, web_cases
from scripts.benchmark_web_routing import classify_light


def test_web_cases_keep_adaptations_distinct_without_duplicating_identical_wording():
    data = {'questions': [
        {'id':'web-01','source_question':'same','prototype_question':'same'},
        {'id':'web-03','source_question':'operational','prototype_question':'schema'},
        {'id':'web-08','source_question':'records','prototype_question':'records'}]}
    cases = web_cases(data)
    assert [(c['case_id'],c['question']) for c in cases] == [
        ('web-01-source','same'),('web-03-source','operational'),
        ('web-03-adapted','schema'),('web-08-source','records')]
    assert cases[-1]['out_of_scope'] and not cases[0]['out_of_scope']


def test_unanswerable_data_request_is_not_scored_as_schema_recall():
    case={'candidate_entities':['Model::Account'], 'out_of_scope':True}
    result=summarize(case,[{'entity_id':'Model::Account'}],[],{},1)
    assert result['candidate_entity_recall'] is None
    assert result['entities']==['Model::Account']


def test_relationship_label_must_exist_in_loaded_graph():
    case={'candidate_entities':['Model::A'], 'out_of_scope':False,
          'candidate_edges':['Model::A --link--> Model::B']}
    with pytest.raises(ValueError,match='Candidate edge not found'):
        summarize(case,[{'entity_id':'Model::A'}],[],{},1)
    result=summarize(case,[{'entity_id':'Model::A'}],['edge1'],
                     {('Model::A','link','Model::B'):'edge1'},1)
    assert result['candidate_edge_recall']==1 and result['candidate_entity_recall']==1


def test_light_classifier_rejects_invalid_output():
    class Fake:
        def chat_json(self,*args,**kwargs): return {'decision':'invented'}
    with pytest.raises(ModelError):
        classify_light('Question',Fake())
