import pytest

from app.ollama import Ollama, chat_context_size


@pytest.mark.parametrize("roles", [
    ("router", "answer", "judge"),
    ("shared", "shared", "other"),
    ("shared", "other", "shared"),
    ("shared", "shared", "shared"),
])
def test_adaptive_override_uses_the_same_cap_for_all_role_aliases(roles):
    router, answer, judge = roles
    client = Ollama("http://localhost", "embed", router, answer, judge_model=judge)
    messages = [{"role": "user", "content": "x" * 18000}]
    client.chat_context_cap_override = 8192
    payload = client._chat_payload(router, messages, stream=True, as_json=True, max_tokens=160, schema={})
    assert payload["options"]["num_ctx"] == 7168
    assert chat_context_size(messages, 160, 8192) == 7168
    longer = [{"role": "user", "content": "x" * 40000}]
    assert client._chat_payload(router, longer, stream=True, as_json=True, max_tokens=160,
                                schema={})["options"]["num_ctx"] == 8192


def test_default_role_dependent_policy_is_unchanged():
    client = Ollama("http://localhost", "embed", "router", "answer", judge_model="judge")
    messages = [{"role": "user", "content": "x" * 18000}]
    def requested(model):
        return client._chat_payload(model, messages, stream=True, as_json=False, max_tokens=160,
                                    schema=None)["options"]["num_ctx"]
    assert requested("router") == 4096
    assert requested("answer") == 7168
    assert requested("judge") == 7168
