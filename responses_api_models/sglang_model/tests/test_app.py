# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp.client_exceptions import ClientResponseError

from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class FakeTokenizer:
    def __init__(self, full_prompt_ids: list[int], decoded: str = "answer") -> None:
        self.full_prompt_ids = full_prompt_ids
        self.decoded = decoded
        self.decode_calls: list[dict] = []
        self.template_calls: list[list[dict]] = []

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
        self.template_calls.append(deepcopy(messages))
        if tokenize:
            has_continuation = (
                any(message.get("role") == "assistant" for message in messages)
                and messages[-1].get("role") != "assistant"
            )
            if has_continuation:
                continued = [40, 41, 90, 91, 30]
                return [*continued, 31] if add_generation_prompt else continued
            if not add_generation_prompt:
                return [40, 41, 90, 91]
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
    def __init__(self, result: dict | BaseException) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def create_generate(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class CustomSuffixTokenizer(FakeTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool):
        if text == "<turn_end>\n":
            assert add_special_tokens is False
            return {"input_ids": [92, 93]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


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
    return json.loads(value)


def test_message_normalization_does_not_inject_rendering_fields() -> None:
    tokenizer = FakeTokenizer([1])
    model = make_model(tokenizer=tokenizer)

    model._full_sglang_tokenize(
        [{"role": "user", "content": "inspect"}],
        tools=None,
        chat_template_kwargs={},
    )

    assert tokenizer.template_calls == [[{"role": "user", "content": "inspect"}]]


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
        assistant_message={"role": "assistant", "content": "a decode that must not be re-tokenized"},
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
        assistant_message={"role": "assistant", "content": "answer"},
    )

    assert model._sglang_session_seq["session-eos"]["seq"] == expected_sequence


def test_session_cache_overlap_never_consumes_prompt_tokens() -> None:
    model = make_model(tokenizer=FakeTokenizer([90]))

    model._update_sglang_session_seq(
        "session-eos-prompt",
        [{"role": "user", "content": "first"}],
        prompt_token_ids=[90],
        generation_token_ids=[],
        tools=None,
        chat_template_kwargs={},
        assistant_message={"role": "assistant", "content": None},
    )

    assert model._sglang_session_seq["session-eos-prompt"]["seq"] == [90, 90, 91]


class NonSpliceableTokenizer(FakeTokenizer):
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
            raise ValueError("template cannot render a stable continuation")
        return "FULL" if add_generation_prompt else "BASE"


class TokenBoundaryTokenizer(FakeTokenizer):
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
            return [1, 9] if add_generation_prompt else [1, 2]
        return "BASESUFFIX" if add_generation_prompt else "BASE"

    def __call__(self, text: str, *, add_special_tokens: bool):
        if text == "SUFFIX":
            return {"input_ids": [9]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


class MismatchedTurnBoundaryTokenizer(FakeTokenizer):
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
            return [1, 2, 3] if add_generation_prompt else [1, 2]
        return "BASESUFFIX" if add_generation_prompt else "BASE"


class DropsContinuationTokenizer(FakeTokenizer):
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
            return [1, 90, 91, 4] if add_generation_prompt else [1, 90, 91]
        return "BASEHEADER" if add_generation_prompt else "BASE"


@pytest.mark.parametrize(
    "tokenizer",
    [
        NonSpliceableTokenizer([700, 701]),
        TokenBoundaryTokenizer([700, 701]),
        MismatchedTurnBoundaryTokenizer([700, 701]),
        DropsContinuationTokenizer([700, 701]),
    ],
)
def test_cached_session_rejects_unprovable_token_splice(tokenizer: FakeTokenizer) -> None:
    model = make_model(tokenizer=tokenizer)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-unspliceable"})
    cached_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "cached"},
    ]
    model._sglang_session_seq["session-unspliceable"] = {
        "messages": deepcopy(cached_messages),
        "seq": [700, 701, 800, 90, 91],
        "rendering_sig": model._sglang_rendering_sig(None, {}),
    }

    with pytest.raises(RuntimeError, match="splice"):
        model._build_sglang_prompt_ids(
            request,
            [*cached_messages, {"role": "user", "content": "continue"}],
            tools=None,
            chat_template_kwargs={},
        )


def test_message_matching_includes_all_rendering_fields() -> None:
    left = [{"role": "tool", "content": "ok", "tool_call_id": "call-1", "name": "shell"}]
    right = [{"role": "tool", "content": "ok", "tool_call_id": "call-2", "name": "shell"}]

    assert not SGLangModel._sglang_messages_match(left, right)


def test_message_matching_ignores_only_training_metadata() -> None:
    left = [
        {
            "role": "assistant",
            "content": "answer",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
            "routed_experts": [[[0]]],
        }
    ]
    right = [{"role": "assistant", "content": "answer"}]

    assert SGLangModel._sglang_messages_match(left, right)


def test_cached_session_rejects_changed_generated_assistant() -> None:
    tokenizer = FakeTokenizer([700, 701])
    model = make_model(tokenizer=tokenizer)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-assistant-mismatch"})
    cached_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "original answer"},
    ]
    model._sglang_session_seq["session-assistant-mismatch"] = {
        "messages": deepcopy(cached_messages),
        "seq": [700, 701, 800, 90, 91],
        "rendering_sig": model._sglang_rendering_sig(None, {}),
    }

    with pytest.raises(RuntimeError, match="history"):
        model._build_sglang_prompt_ids(
            request,
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "different answer"},
                {"role": "user", "content": "continue"},
            ],
            tools=None,
            chat_template_kwargs={},
        )


async def test_text_followup_survives_responses_converter_round_trip() -> None:
    tokenizer = FakeTokenizer([1], decoded="first answer")
    client = FakeSGLangClient(
        {
            "text": "first answer",
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            },
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)
    request = SimpleNamespace(session={SESSION_ID_KEY: "responses-round-trip"})
    first_request = NeMoGymResponseCreateParamsNonStreaming(input="first question")
    first_chat_request = model._converter.responses_to_chat_completion_create_params(first_request)
    first_chat_response = await model.chat_completions(request, first_chat_request)
    first_output = model._converter.postprocess_chat_response(first_chat_response.choices[0])
    second_request = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "first question"}],
            },
            *[item.model_dump() for item in first_output],
            {"role": "user", "content": "continue"},
        ]
    )
    second_chat_request = model._converter.responses_to_chat_completion_create_params(second_request)

    followup_prompt, session_id = model._build_sglang_prompt_ids(
        request,
        second_chat_request.model_dump(exclude_unset=True)["messages"],
        tools=None,
        chat_template_kwargs={},
    )

    assert session_id == "responses-round-trip"
    assert followup_prompt == [1, 11, 90, 91, 30, 31]


def test_stale_session_commit_is_rejected_without_mutating_newer_state() -> None:
    model = make_model(tokenizer=FakeTokenizer([1]))
    stale_state = {"messages": [], "seq": [1], "rendering_sig": model._sglang_rendering_sig(None, {})}
    newer_state = {
        "messages": [{"role": "assistant", "content": "newer"}],
        "seq": [9],
        "rendering_sig": model._sglang_rendering_sig(None, {}),
    }
    model._sglang_session_seq["concurrent-session"] = newer_state

    with pytest.raises(RuntimeError, match="changed while a continuation was in flight"):
        model._update_sglang_session_seq(
            "concurrent-session",
            [{"role": "user", "content": "stale"}],
            prompt_token_ids=[1],
            generation_token_ids=[2],
            tools=None,
            chat_template_kwargs={},
            assistant_message={"role": "assistant", "content": "stale"},
            expected_state=stale_state,
        )

    assert model._sglang_session_seq["concurrent-session"] is newer_state


async def test_aborted_generation_does_not_mutate_session_cache() -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    client = FakeSGLangClient(
        {
            "text": "partial",
            "meta_info": {
                "output_ids": [10],
                "output_token_logprobs": [{"token_id": 10, "logprob": -0.1}],
                "finish_reason": {"type": "abort"},
            },
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)
    before = deepcopy(model._sglang_session_seq)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
    )

    with pytest.raises(RuntimeError, match="finish_reason='abort'"):
        await model.chat_completions(
            SimpleNamespace(session={SESSION_ID_KEY: "abort-session"}),
            body,
        )

    assert model._sglang_session_seq == before


def test_hermes_parser_uses_parsed_arguments_regardless_of_field_order() -> None:
    model = make_model()

    _, content, tool_calls = model._parse_sglang_generation(
        '<tool_call>{"arguments":{"outer":{"x":1}},"name":"shell"}</tool_call>'
    )

    assert content == ""
    assert tool_calls[0]["function"]["name"] == "shell"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"outer": {"x": 1}}


def test_hermes_parser_preserves_malformed_blocks_as_visible_content() -> None:
    model = make_model()
    malformed = '<tool_call>{"name":"broken","arguments":not-json}</tool_call>'
    valid = '<tool_call>{"name":"shell","arguments":{"x":1}}</tool_call>'

    _, content, tool_calls = model._parse_sglang_generation(f"before {malformed} middle {valid} after")

    assert malformed in content
    assert content.startswith("before")
    assert content.endswith("after")
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "shell"


def make_client_response_error(status: int, payload: object) -> ClientResponseError:
    error = ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=status,
        message="SGLang request failed",
    )
    error.response_content = json.dumps(payload).encode()
    return error


@pytest.mark.parametrize(
    "message",
    [
        "The input (65 tokens) is longer than the model's context length (64 tokens).",
        (
            "Requested token count exceeds the model's maximum context length of 64 tokens. "
            "You requested a total of 65 tokens: 60 tokens from the input messages and 5 tokens "
            "for the completion. Please reduce the number of tokens in the input messages or the "
            "completion to fit within the limit."
        ),
    ],
)
async def test_structured_bad_request_context_error_returns_length(message: str) -> None:
    model = make_model(
        tokenizer=FakeTokenizer([1, 2, 3]),
        client=FakeSGLangClient(make_client_response_error(400, {"error": {"message": message}})),
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "x"}])

    response = await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "context-error"}),
        body,
    )

    assert response.choices[0].finish_reason == "length"


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (500, {"error": {"message": "The input (65 tokens) is longer than the model's context length (64 tokens)."}}),
        (400, {"error": {"message": "unrelated longer than server failure"}}),
        (400, {"message": "missing native error envelope"}),
    ],
)
async def test_unrelated_http_errors_are_not_reported_as_context_length(status: int, payload: object) -> None:
    error = make_client_response_error(status, payload)
    model = make_model(
        tokenizer=FakeTokenizer([1, 2, 3]),
        client=FakeSGLangClient(error),
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "x"}])

    with pytest.raises(ClientResponseError) as raised:
        await model.chat_completions(
            SimpleNamespace(session={SESSION_ID_KEY: "server-error"}),
            body,
        )

    assert raised.value is error


async def test_sampling_controls_merge_with_explicit_request_precedence() -> None:
    tokenizer = FakeTokenizer([1])
    client = FakeSGLangClient(
        {
            "text": "answer",
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            },
        }
    )
    model = make_model(
        tokenizer=tokenizer,
        client=client,
        extra_body={"seed": 1, "top_k": 20, "repetition_penalty": 1.1},
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        metadata={"extra_body": json.dumps({"seed": 2, "top_k": 10, "min_p": 0.2})},
        seed=3,
        frequency_penalty=0.4,
        presence_penalty=0.5,
        logit_bias={"11": 1},
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "sampling-controls"}),
        body,
    )

    sampling_params = client.calls[0]["sampling_params"]
    assert sampling_params["sampling_seed"] == 3
    assert sampling_params["top_k"] == 10
    assert sampling_params["min_p"] == 0.2
    assert sampling_params["repetition_penalty"] == 1.1
    assert sampling_params["frequency_penalty"] == 0.4
    assert sampling_params["presence_penalty"] == 0.5
    assert sampling_params["logit_bias"] == {"11": 1}


async def test_metadata_rollout_seed_reaches_native_sampling_seed() -> None:
    tokenizer = FakeTokenizer([1])
    client = FakeSGLangClient(
        {
            "text": "answer",
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            },
        }
    )
    model = make_model(tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        metadata={"extra_body": '{"seed":42}'},
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "metadata-seed"}),
        body,
    )

    assert client.calls[0]["sampling_params"]["sampling_seed"] == 42


@pytest.mark.parametrize("max_field", ["max_tokens", "max_completion_tokens"])
async def test_explicit_null_max_tokens_clears_configured_limit(max_field: str) -> None:
    tokenizer = FakeTokenizer([1])
    client = FakeSGLangClient(
        {
            "text": "answer",
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            },
        }
    )
    model = make_model(
        context_length=64,
        tokenizer=tokenizer,
        client=client,
        extra_body={"max_new_tokens": 37},
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        **{max_field: None},
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: f"null-{max_field}"}),
        body,
    )

    assert client.calls[0]["sampling_params"]["max_new_tokens"] == 55


async def test_nonnull_legacy_max_tokens_wins_over_null_max_completion_tokens() -> None:
    tokenizer = FakeTokenizer([1])
    client = FakeSGLangClient(
        {
            "text": "answer",
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            },
        }
    )
    model = make_model(context_length=64, tokenizer=tokenizer, client=client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        max_completion_tokens=None,
        max_tokens=7,
    )

    await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "dual-max"}),
        body,
    )

    assert client.calls[0]["sampling_params"]["max_new_tokens"] == 7


async def test_conflicting_extra_body_aliases_fail_closed() -> None:
    model = make_model(
        tokenizer=FakeTokenizer([1]),
        client=FakeSGLangClient({}),
        extra_body={"seed": 1, "sampling_seed": None},
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "x"}])

    with pytest.raises(ValueError, match="conflicting aliases"):
        await model.chat_completions(
            SimpleNamespace(session={SESSION_ID_KEY: "conflicting-aliases"}),
            body,
        )


@pytest.mark.parametrize(
    "body_kwargs",
    [
        {"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "shell"}}]},
        {"parallel_tool_calls": False, "tools": [{"type": "function", "function": {"name": "shell"}}]},
        {"n": 2},
    ],
)
async def test_unsupported_generation_controls_fail_closed(body_kwargs: dict) -> None:
    model = make_model(tokenizer=FakeTokenizer([1]), client=FakeSGLangClient({}))
    body = NeMoGymChatCompletionCreateParamsNonStreaming(
        messages=[{"role": "user", "content": "inspect"}],
        **body_kwargs,
    )

    with pytest.raises((NotImplementedError, ValueError), match="SGLang"):
        await model.chat_completions(
            SimpleNamespace(session={SESSION_ID_KEY: "unsupported-controls"}),
            body,
        )


async def test_unknown_extra_body_control_fails_closed() -> None:
    model = make_model(
        tokenizer=FakeTokenizer([1]),
        client=FakeSGLangClient({}),
        extra_body={"unknown_sampling_control": 1},
    )
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "x"}])

    with pytest.raises(ValueError, match="unknown_sampling_control"):
        await model.chat_completions(
            SimpleNamespace(session={SESSION_ID_KEY: "unknown-control"}),
            body,
        )


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
        assistant_message={"role": "assistant", "content": "cached"},
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


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"sglang_eos_markers": [""]},
        {"sglang_turn_suffix": ""},
    ],
)
def test_generate_transport_rejects_empty_turn_markers(config_overrides: dict) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        make_model(transport="generate", **config_overrides)


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
async def test_chat_attach_rejects_visible_output_with_empty_training_metadata() -> None:
    model = make_model(transport="chat", context_length=None)
    choice = _chat_choice(meta_info={"output_token_logprobs": []})

    with pytest.raises(RuntimeError, match="empty generated-token metadata"):
        await model._attach_token_id_information(choice, {}, MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_token_id", [True, "1", 1.0])
async def test_chat_attach_rejects_malformed_prompt_token_ids(prompt_token_id: object) -> None:
    model = make_model(transport="chat", context_length=None)
    choice = _chat_choice(prompt_token_ids=[prompt_token_id])

    with pytest.raises(RuntimeError, match="prompt_token_ids token ID"):
        await model._attach_token_id_information(choice, {}, MagicMock())


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
async def test_chat_attach_prioritizes_transport_diagnostic_when_all_extensions_are_missing() -> None:
    model = make_model(transport="chat", context_length=None)
    choice = _chat_choice(meta_info={})
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
            SimpleNamespace(session={SESSION_ID_KEY: "aborted-generation"}),
            {"messages": [{"role": "user", "content": "hi"}]},
        )


def test_generate_transport_uses_configured_turn_suffix_in_cache_and_signature() -> None:
    model = make_model(
        tokenizer=CustomSuffixTokenizer([1]),
        sglang_turn_suffix="<turn_end>\n",
    )

    model._update_sglang_session_seq(
        "custom-suffix",
        [{"role": "user", "content": "first"}],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        tools=None,
        chat_template_kwargs={},
        assistant_message={"role": "assistant", "content": "answer"},
    )

    assert model._sglang_session_seq["custom-suffix"]["seq"] == [1, 2, 92, 93]
    assert model._sglang_session_seq["custom-suffix"]["rendering_sig"][-1] == "<turn_end>\n"


async def test_generate_transport_strips_configured_eos_marker() -> None:
    tokenizer = FakeTokenizer([1], decoded="answer<END>")
    client = FakeSGLangClient(
        {
            "meta_info": {
                "output_ids": [11],
                "output_token_logprobs": [{"token_id": 11, "logprob": -0.1}],
            }
        }
    )
    model = make_model(
        tokenizer=tokenizer,
        client=client,
        sglang_eos_markers=["<END>"],
    )

    response = await model.chat_completions(
        SimpleNamespace(session={SESSION_ID_KEY: "custom-eos-marker"}),
        NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "inspect"}]),
    )

    assert response.choices[0].message.content == "answer"
