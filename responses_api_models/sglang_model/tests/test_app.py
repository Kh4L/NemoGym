# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class FakeTokenizer:
    def __init__(self, full_prompt_ids: list[int], decoded: str = "answer") -> None:
        self.full_prompt_ids = full_prompt_ids
        self.decoded = decoded
        self.decode_calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        chat_template=None,
        add_generation_prompt,
        tokenize,
        **kwargs,
    ):
        if tokenize:
            return list(self.full_prompt_ids)
        if len(messages) == 1 and messages[0] == {"role": "assistant", "content": "X"}:
            assert add_generation_prompt is False
            return "ANCHOR"
        assert add_generation_prompt is True
        return "ANCHORFOLLOWUP"

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        if text == "<|im_end|>\n":
            return {"input_ids": [90, 91]}
        if text == "FOLLOWUP":
            return {"input_ids": [30, 31]}
        raise AssertionError(f"unexpected tokenization input: {text!r}")

    def decode(self, token_ids, *, skip_special_tokens: bool, spaces_between_special_tokens: bool):
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
            }
        )
        return self.decoded


class FakeSGLangClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def create_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def make_model(
    *,
    context_length: int = 64,
    tokenizer: FakeTokenizer | None = None,
    client: FakeSGLangClient | None = None,
    transport: str = "generate",
    **config_overrides,
) -> SGLangModel:
    config = SGLangModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sglang_model",
        base_url="http://localhost:30000/v1",
        api_key="unused",  # pragma: allowlist secret
        model="local-tokenizer",
        transport=transport,
        context_length=context_length,
        return_token_id_information=True,
        uses_reasoning_parser=True,
        **config_overrides,
    )
    model = SGLangModel(
        config=config,
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    if tokenizer is not None:
        model._sglang_tokenizer = tokenizer
    if client is not None:
        model._clients = [client]
    return model


async def test_generate_path_preserves_training_ids_reasoning_and_tools() -> None:
    tokenizer = FakeTokenizer(
        [1, 2, 3],
        decoded=(
            'private reasoning</think><tool_call>{"name":"shell","arguments":{"command":"ls"}}</tool_call><|im_end|>'
        ),
    )
    client = FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": [11, 12],
                "output_token_logprobs": [
                    {"token_id": 11, "logprob": -0.1},
                    {"id": 12, "logprob": -0.2},
                ],
                "finish_reason": {"type": "stop"},
            }
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        max_tokens=8,
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-1"})

    response = await model.chat_completions(request, body)

    assert client.calls == [
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "spaces_between_special_tokens": False,
                "max_new_tokens": 8,
            },
            "return_logprob": True,
        }
    ]
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "<think>private reasoning</think>"
    assert choice.message.prompt_token_ids == [1, 2, 3]
    assert choice.message.generation_token_ids == [11, 12]
    assert choice.message.generation_log_probs == [-0.1, -0.2]
    assert choice.message.tool_calls[0].function.name == "shell"
    assert json_loads(choice.message.tool_calls[0].function.arguments) == {"command": "ls"}
    assert tokenizer.decode_calls == [
        {
            "token_ids": [11, 12],
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    ]


def json_loads(value: str) -> dict:
    import json

    return json.loads(value)


def test_followup_prompt_splices_exact_sampled_ids() -> None:
    tokenizer = FakeTokenizer([10, 11])
    model = make_model(tokenizer=tokenizer)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-2"})
    first_messages = [{"role": "user", "content": "first"}]

    first_prompt, session_id = model._build_sglang_prompt_ids(
        request,
        first_messages,
        tools=None,
        chat_template_kwargs={},
    )
    model._update_sglang_session_seq(
        session_id,
        first_messages,
        first_prompt,
        generation_token_ids=[20, 21],
        tools=None,
        chat_template_kwargs={},
    )
    followup_messages = [
        *first_messages,
        {"role": "assistant", "content": "a decode that must not be re-tokenized"},
        {"role": "user", "content": "continue"},
    ]

    followup_prompt, _ = model._build_sglang_prompt_ids(
        request,
        followup_messages,
        tools=None,
        chat_template_kwargs={},
    )

    assert followup_prompt == [10, 11, 20, 21, 90, 91, 30, 31]


@pytest.mark.parametrize(
    ("generation_token_ids", "expected_sequence"),
    [
        ([20], [1, 20, 90, 91]),
        ([20, 90], [1, 20, 90, 91]),
        ([20, 90, 91], [1, 20, 90, 91]),
    ],
)
def test_session_cache_appends_only_missing_eos_suffix(
    generation_token_ids: list[int],
    expected_sequence: list[int],
) -> None:
    model = make_model(tokenizer=FakeTokenizer([1]))

    model._update_sglang_session_seq(
        "session-eos",
        [{"role": "user", "content": "first"}],
        prompt_token_ids=[1],
        generation_token_ids=generation_token_ids,
        tools=None,
        chat_template_kwargs={},
    )

    assert model._sglang_session_seq["session-eos"]["seq"] == expected_sequence


@pytest.mark.parametrize("max_tokens_field", ["max_completion_tokens", "max_tokens"])
async def test_explicit_max_tokens_is_clamped_to_remaining_context(
    max_tokens_field: str,
) -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    client = FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [
                    {"token_id": 11, "logprob": -0.1},
                ],
            }
        }
    )
    model = make_model(
        context_length=10,
        tokenizer=tokenizer,
        client=client,
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        **{max_tokens_field: 10},
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "session-clamp"}),
        body,
    )

    assert client.calls[0]["sampling_params"]["max_new_tokens"] == 7


async def test_over_context_terminates_without_calling_generate() -> None:
    tokenizer = FakeTokenizer([1, 2, 3, 4])
    client = FakeSGLangClient({"must": "not be used"})
    model = make_model(context_length=4, tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "too long"}],
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-3"})

    response = await model.chat_completions(request, body)

    assert client.calls == []
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.prompt_token_ids == [1, 2, 3, 4]
    assert response.choices[0].message.generation_token_ids == []


def test_followup_prompt_rejects_changed_rendering_inputs() -> None:
    tokenizer = FakeTokenizer([10, 11])
    model = make_model(tokenizer=tokenizer)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-rendering"})
    first_messages = [{"role": "user", "content": "first"}]
    first_prompt, session_id = model._build_sglang_prompt_ids(
        request,
        first_messages,
        tools=[{"type": "function", "function": {"name": "old"}}],
        chat_template_kwargs={"enable_thinking": True},
    )
    model._update_sglang_session_seq(
        session_id,
        first_messages,
        first_prompt,
        generation_token_ids=[20, 21],
        tools=[{"type": "function", "function": {"name": "old"}}],
        chat_template_kwargs={"enable_thinking": True},
    )
    followup_messages = [
        *first_messages,
        {"role": "assistant", "content": "cached"},
        {"role": "user", "content": "continue"},
    ]

    with pytest.raises(RuntimeError, match="session tools or chat-template"):
        model._build_sglang_prompt_ids(
            request,
            followup_messages,
            tools=[{"type": "function", "function": {"name": "new"}}],
            chat_template_kwargs={"enable_thinking": True},
        )
    with pytest.raises(RuntimeError, match="session tools or chat-template"):
        model._build_sglang_prompt_ids(
            request,
            followup_messages,
            tools=[{"type": "function", "function": {"name": "old"}}],
            chat_template_kwargs={"enable_thinking": False},
        )


def test_sglang_config_owns_context_and_tool_format() -> None:
    model = make_model(context_length=128)

    assert model.config.context_length == 128
    assert model.config.sglang_tool_format == "hermes"
    assert not hasattr(model.config, "engine")


def test_inline_chat_template_is_used_directly() -> None:
    model = make_model()
    model.config.sglang_chat_template = "inline-template"
    model.config.sglang_chat_template_path = "/must/not/be/read"

    assert model._get_sglang_chat_template() == "inline-template"


# ---------------------------------------------------------------------------
# transport="chat": inherits the vLLM path, overrides only token extraction
# ---------------------------------------------------------------------------


def _chat_choice(**overrides):
    """An SGLang >= 0.5.13 chat choice with the native TITO extensions populated."""
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "the answer"},
        "prompt_token_ids": [1, 2, 3],
        "meta_info": {"output_token_logprobs": [[-0.5, 10, "a"], [-0.25, 11, "b"]]},
        "logprobs": {"content": [{"token": "a", "logprob": -0.5}]},
    }
    choice.update(overrides)
    return choice


def test_chat_transport_needs_no_context_length() -> None:
    """SGLang enforces its own window on the chat path, so the knob is generate-only."""
    model = make_model(transport="chat", context_length=None)
    assert model.config.context_length is None


def test_generate_transport_requires_context_length() -> None:
    """A locally tokenized prompt is unbounded unless this server bounds it."""
    with pytest.raises(ValueError, match="context_length is required"):
        make_model(transport="generate", context_length=None)


def test_chat_preprocess_requests_sglang_tito_extensions() -> None:
    model = make_model(transport="chat", context_length=None)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    out = model._preprocess_chat_completion_create_params(MagicMock(), body)

    assert out["return_meta_info"] is True
    assert out["return_prompt_token_ids"] is True
    # vLLM's `token_id:NNN` encoding does not exist on SGLang; leaving it set would be a
    # silent no-op that misrepresents where the ids come from.
    assert "return_tokens_as_token_ids" not in out
    # ...and the inherited behavior is still in force.
    assert out["logprobs"] is True
    assert out["model"] == "local-tokenizer"


def test_chat_preprocess_honors_per_request_chat_template_kwargs() -> None:
    """Per-sample overrides must survive; dropping them renders the wrong template."""
    model = make_model(transport="chat", context_length=None, chat_template_kwargs={"enable_thinking": False})
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"chat_template_kwargs": '{"enable_thinking": true}'},
    }

    out = model._preprocess_chat_completion_create_params(MagicMock(), body)

    assert out["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.asyncio
async def test_chat_attach_reads_native_ids_and_logprobs() -> None:
    model = make_model(transport="chat", context_length=None)
    choice = _chat_choice()

    await model._attach_token_id_information(choice, {}, MagicMock())

    assert choice["message"]["prompt_token_ids"] == [1, 2, 3]
    assert choice["message"]["generation_token_ids"] == [10, 11]
    assert choice["message"]["generation_log_probs"] == [-0.5, -0.25]
    # Non-OpenAI / duplicated fields are stripped so the response validates.
    for key in ("logprobs", "prompt_token_ids", "meta_info"):
        assert key not in choice


@pytest.mark.asyncio
async def test_chat_attach_rejects_aborted_generation() -> None:
    """An abort is a truncated fragment; it must not enter a batch as a `stop`."""
    model = make_model(transport="chat", context_length=None)

    with pytest.raises(RuntimeError, match="abort"):
        await model._attach_token_id_information(_chat_choice(finish_reason="abort"), {}, MagicMock())


@pytest.mark.asyncio
async def test_chat_attach_errors_when_server_predates_chat_tito() -> None:
    """Older SGLang ignores the extensions; say so instead of emitting empty token ids."""
    model = make_model(transport="chat", context_length=None)
    choice = _chat_choice()
    choice.pop("prompt_token_ids")

    with pytest.raises(RuntimeError, match="0.5.13"):
        await model._attach_token_id_information(choice, {}, MagicMock())


@pytest.mark.asyncio
async def test_generate_transport_rejects_aborted_generation() -> None:
    """Same guarantee on the /generate path, where SGLang reports abort in meta_info."""
    tokenizer = FakeTokenizer(full_prompt_ids=[1, 2, 3])
    client = FakeSGLangClient(
        {
            "meta_info": {
                "finish_reason": {"type": "abort"},
                "output_token_logprobs": [[-0.5, 10, "a"]],
            }
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)

    with pytest.raises(RuntimeError, match="abort"):
        await model._sglang_chat_completion(
            SimpleNamespace(session={}),
            {"messages": [{"role": "user", "content": "hi"}]},
        )
