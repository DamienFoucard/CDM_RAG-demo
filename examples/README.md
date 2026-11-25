# API examples

These examples use the included terminal client to call a hosted CDM RAG
service. The outputs below are the **actual final answers** from two requests
to the independent GPU service on 25 November 2026. The answers and citations
can differ between runs; they are examples, not reference answers.

## Setup

From the repository root, put the service host and your application token in a
private `.env` file. Replace both placeholders with the values you were given;
do not commit this file or put a real token in a command or URL.

```text
CDM_RAG_GPU_PUBLIC_HOST=cdm-rag.example.com
CDM_RAG_API_TOKEN=your-application-token
```

Check the token without asking a question:

```sh
python3 scripts/check_api_token.py --env-file .env
```

The client displays progress and model calls on stderr and prints the final
answer to stdout. The output blocks below show the final answer only. Add
`--json` before the question to see the full response, source passages, and
processing metadata. Model loading and inference can take several minutes.
Questions and answers are saved in the service's query log.

## Easy: overdraft definition

Input:

```sh
python3 scripts/ask_cdm.py --env-file .env \
  'What does FIOverdraft describe?'
```

Output from one run:

```text
FIOverdraft is a CDM entity in the RetailBankingCoreDataModel that represents an extension of credit associated with a checking account, allowing the account holder to continue withdrawing funds when the account reaches zero [RetailBankingCoreDataModel::FIOverdraft · R2]. It extends the base CdmEntity and includes attributes such as fi_OverdraftId for unique identification and statecode/statuscode for tracking status [RetailBankingCoreDataModel::FIOverdraft · R4]. The entity also exhibits traits like is.localized.describedAs with a description of its function and is.CDM.entityVersion indicating version 1.1 [RetailBankingCoreDataModel::FIOverdraft · R3].
```

The cited passages point to the pinned [FIOverdraft definition](https://github.com/microsoft/CDM/blob/dd21d715e05ebf740a11356c80b5c3b4c38a89c2/schemaDocuments/FinancialServices/RetailBankingCoreDataModel/Fi_overdraft.cdm.json).
The definition and attributes support the main description, but the cited R4
passage does not itself establish the inheritance claim. This illustrates why
a citation should be checked against the actual passage, not just its label.

## Challenging: customer-to-holding chain

Input:

```sh
python3 scripts/ask_cdm.py --env-file .env \
  'A bank needs the schema links from a person or company to something they own and onward to its specific loan or investment type. Which links establish this chain?'
```

Output from one run:

```text
The provided passages describe attributes of entities such as FHLoan, Relationship, and Account but do not contain schema links connecting a person or company to the assets they own. Consequently, there is no evidence in the supplied text to establish a chain linking a customer to their specific loan or investment types [RetailBankingCoreDataModel::FHLoan · R1][RetailBankingCoreDataModel::FHLoan · R2][RetailBankingCoreDataModel::Relationship][applicationCommon::Account].
```

In that run, retrieval returned attribute passages for [FHLoan](https://github.com/microsoft/CDM/blob/dd21d715e05ebf740a11356c80b5c3b4c38a89c2/schemaDocuments/FinancialServices/RetailBankingCoreDataModel/Fh_loan.cdm.json), [Relationship](https://github.com/microsoft/CDM/blob/dd21d715e05ebf740a11356c80b5c3b4c38a89c2/schemaDocuments/FinancialServices/RetailBankingCoreDataModel/Relationship.cdm.json), and [Account](https://github.com/microsoft/CDM/blob/dd21d715e05ebf740a11356c80b5c3b4c38a89c2/schemaDocuments/core/applicationCommon/Account.cdm.json), but missed the relationship passages needed for the chain. The answer correctly describes that lack of supplied evidence; **this run did not solve the multi-hop question**. See the [development challenge](../evaluation/reformulation-challenge-candidate-v1.json) for its expected passage targets.

For event types, authoritative final results and metadata fields, see the
[streaming API guide](../docs/api-streaming.md).
