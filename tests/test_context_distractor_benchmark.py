import pytest

from scripts.benchmark_context_distractors import build_cases, messages_for, score, select_cases


def corpus():
    proofs = [
        ("paired-dev-card-customer", "rewrite", "R3", "CDM entity FICard has a relationship through Cardholder to RetailBankingCoreDataModel::Account, RetailBankingCoreDataModel::Contact."),
        ("paired-dev-common-account-contact", "raw", "R2", "foundationCommon Contact.parentCustomerId maps to Account.cdm.json/Account and Contact.cdm.json/Contact."),
        ("paired-test-customer-partner", "rewrite", "R3", "foundationCommon CustomerRelationship.customerId and CustomerRelationship.partnerId map to Account.cdm.json/Account and Contact.cdm.json/Contact."),
    ]
    cases = [{"case_id":case, "arm":arm, "sources":[{"citation":label,"text":text,"passage_id":case}]}
             for case,arm,label,text in proofs]
    # Separate original passages: no repetition or synthetic answer content.
    cases.append({"case_id":"other", "arm":"raw", "sources":[
        {"citation":f"R{i}", "text":f"Entity extra_{i}. " + chr(65+i)*500,"passage_id":f"extra_{i}"} for i in range(4)
    ]})
    return {"cases":cases}


def test_position_moves_only_proof_and_preserves_source_evidence():
    cases=build_cases(corpus(),1000)
    assert len(cases)==9
    for i in range(0,len(cases),3):
        early,middle,late=cases[i:i+3]
        assert [x["position"] for x in (early,middle,late)]==["early","middle","late"]
        proof=early["expected_citation"]
        assert early["evidence"][0]["label"] == proof
        assert late["evidence"][-1]["label"] == proof
        assert sorted(x["label"] for x in early["evidence"])==sorted(x["label"] for x in middle["evidence"])==sorted(x["label"] for x in late["evidence"])
        assert [x["label"] for x in early["evidence"] if x["label"] != proof] == [x["label"] for x in late["evidence"] if x["label"] != proof]


def test_gold_is_not_injected_and_scoring_requires_exact_set_and_citation():
    case=build_cases(corpus(),1000)[0]
    messages=messages_for(case)
    assert "expected_items" not in str(messages) and "expected_citation" not in str(messages)
    correct={"items":["contact","Account"],"citation":case["expected_citation"]}
    assert score(case,correct)["fully_correct"]
    assert not score(case,{**correct,"items":["Account"]})["fully_correct"]
    assert not score(case,{**correct,"items":["Account","Contact","Group"]})["fully_correct"]
    assert not score(case,{**correct,"items":["Account","Account","Contact"]})["fully_correct"]
    assert not score(case,{**correct,"citation":"Pwrong"})["fully_correct"]
    with pytest.raises(ValueError): score(case,{"items":"Account", "citation":"Pwrong"})


def test_source_change_fails_before_any_run():
    source=corpus()
    source["cases"][0]["sources"][0]["text"]="Changed proof"
    with pytest.raises(ValueError,match="Source changed"):
        build_cases(source,1000)


def test_pilot_selection_is_exact_and_invalid_selection_fails():
    cases = build_cases(corpus(), 1000)
    assert len(select_cases(cases, None, None)) == 9
    selected = select_cases(cases, ["bank-card-targets"], ["late"])
    assert len(selected) == 1
    assert selected[0]["case_id"] == "bank-card-targets" and selected[0]["position"] == "late"
    with pytest.raises(ValueError): select_cases(cases, ["invalid"], None)
    with pytest.raises(ValueError): select_cases(cases, None, ["late", "late"])
