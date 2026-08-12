# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym model server for SGLang-served policies, preserving exact sampled token IDs.

Two transports, both attaching `prompt_token_ids` / `generation_token_ids` /
`generation_log_probs` to the assistant message for RL training:

``transport: chat`` (default, **requires sglang >= 0.5.13**)
    Drives SGLang's OpenAI-compatible ``/v1/chat/completions`` and asks for the training
    metadata via the native ``return_meta_info`` / ``return_prompt_token_ids`` request
    extensions (added by the TITO sync series; present in the 0.5.13 release tree -- no
    patched build or fork required). Everything except the token extraction is inherited
    from ``VLLMModel``, so prompt templating, tool-call parsing, sampling params, auth and
    context-overflow handling are all done *server-side*. There is no local tokenizer that
    can drift from the server's, and no client-side parser to keep in sync.

``transport: generate``
    Drives SGLang's native ``/generate`` with ``return_logprob=True``, rendering and
    tokenizing the prompt locally. Required for builds/forks that predate chat-side TITO
    (e.g. those serving diffusion LLMs). This path additionally keeps **multi-turn prompts
    prefix-stable** by splicing the previous turn's exact sampled ids instead of
    re-tokenizing history, and parses tool calls client-side since ``/generate`` returns
    raw text.

The pure request/response transforms live in `_logic.py`; tool-call parsing in
`tool_parsers.py`. Both are unit-tested without a live server.
"""

import json
import re
from copy import deepcopy
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import Field, model_validator

from nemo_gym.base_responses_api_model import Body
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_models.sglang_model._logic import (
    _normalize_token_ids,
    extract_generated_tokens_and_logprobs,
)
from responses_api_models.sglang_model.tool_parsers import (
    normalize_tool_call_arguments,
    parse_qwen3_coder_tool_calls,
)
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


# Chat-side TITO (`return_meta_info` / `return_prompt_token_ids` on /v1/chat/completions)
# landed in the sglang-miles sync series and is present in the 0.5.13 release tree.
MIN_SGLANG_VERSION_FOR_CHAT_TRANSPORT = "0.5.13"


class SGLangModelConfig(VLLMModelConfig):
    """Configuration for exact-token SGLang generation."""

    # `chat` requires sglang >= 0.5.13. `generate` works on any build but tokenizes locally.
    transport: Literal["chat", "generate"] = "chat"

    # --- `generate` transport only (unused when transport == "chat") ---
    # Required for `generate`: the server enforces no limit on a pre-tokenized prompt, so the
    # adapter must. `chat` needs no value -- SGLang applies its own limit and the inherited
    # overflow handling recognizes the error.
    context_length: Optional[int] = Field(default=None, gt=0)
    trust_remote_code: bool = False
    sglang_chat_template: Optional[str] = None
    sglang_chat_template_path: Optional[str] = None
    sglang_tool_format: Literal["hermes", "qwen3_coder"] = "hermes"
    # End-of-turn markers stripped from generated text before parsing, and the end-of-turn
    # sequence appended when splicing a turn into the cached prefix. Defaults are ChatML
    # (Qwen etc.); a model whose template uses different markers MUST override both, or the
    # splice will emit a malformed turn boundary.
    sglang_eos_markers: List[str] = Field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    sglang_turn_suffix: str = "<|im_end|>\n"

    @model_validator(mode="after")
    def _validate_transport_requirements(self) -> "SGLangModelConfig":
        if self.transport == "generate" and self.context_length is None:
            raise ValueError(
                "context_length is required when transport='generate': the prompt is tokenized "
                "locally and sent as input_ids, so this server must enforce the window itself. "
                "Set it to the SGLang server's max total sequence length."
            )
        if self.transport == "generate" and any(not marker for marker in self.sglang_eos_markers):
            raise ValueError("sglang_eos_markers entries must be non-empty when transport='generate'")
        if self.transport == "generate" and not self.sglang_turn_suffix:
            raise ValueError("sglang_turn_suffix must be non-empty when transport='generate'")
        return self


class SGLangModel(VLLMModel):
    """Responses-API adapter that preserves exact sampled token IDs."""

    config: SGLangModelConfig

    _SGLANG_TOOL_CALL_PATTERN: ClassVar = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL,
    )
    _SGLANG_TRAINING_MESSAGE_FIELDS: ClassVar = frozenset(
        {
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
            "routed_experts",
        }
    )
    _SGLANG_NATIVE_SAMPLING_FIELDS: ClassVar = frozenset(
        {
            "max_new_tokens",
            "stop",
            "stop_token_ids",
            "stop_regex",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "min_new_tokens",
            "n",
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "ignore_eos",
            "no_stop_trim",
            "logit_bias",
            "sampling_seed",
            "custom_params",
            "response_format",
        }
    )
    _SGLANG_EXTRA_BODY_ALIASES: ClassVar = {
        "seed": "sampling_seed",
        "min_tokens": "min_new_tokens",
        "max_tokens": "max_new_tokens",
        "max_completion_tokens": "max_new_tokens",
    }
    _SGLANG_ADAPTER_OWNED_SAMPLING_FIELDS: ClassVar = frozenset(
        {
            "spaces_between_special_tokens",
            "skip_special_tokens",
            "stream_interval",
        }
    )
    _SGLANG_GRAMMAR_FIELDS: ClassVar = ("json_schema", "regex", "ebnf", "structural_tag")

    def _post_init(self) -> None:
        super()._post_init()
        self._sglang_tokenizer: Any = None
        self._sglang_chat_template: Optional[str] = None
        self._sglang_session_seq: Dict[str, Dict[str, Any]] = {}
        self._sglang_eos_nl_ids: Optional[List[int]] = None

    def _get_sglang_tokenizer(self) -> Any:
        if self._sglang_tokenizer is None:
            from transformers import AutoTokenizer

            self._sglang_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._sglang_tokenizer

    def _get_sglang_chat_template(self) -> Optional[str]:
        if self.config.sglang_chat_template is not None:
            return self.config.sglang_chat_template
        if self._sglang_chat_template is None and self.config.sglang_chat_template_path:
            with open(self.config.sglang_chat_template_path) as template_file:
                self._sglang_chat_template = template_file.read()
        return self._sglang_chat_template

    async def chat_completions(
        self,
        request: Request,
        body: NeMoGymChatCompletionCreateParamsNonStreaming = Body(),
    ) -> NeMoGymChatCompletion:
        """Generate without applying the vLLM-specific request preprocessing.

        That applies to ``transport="generate"``, which needs a pre-tokenized prompt and so
        renders locally. ``transport="chat"`` instead inherits the whole vLLM path and
        differs only in token extraction -- see ``_attach_token_id_information``.
        """
        if self.config.transport == "chat":
            return await super().chat_completions(request, body)
        return await self._sglang_chat_completion(
            request,
            body.model_dump(exclude_unset=True),
        )

    # ------------------------------------------------------------------
    # `chat` transport: inherit everything, override only token extraction
    # ------------------------------------------------------------------

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        if self.config.transport == "chat" and self.config.return_token_id_information:
            # vLLM-only knob: SGLang has no `token_id:NNN` logprob encoding, and leaving it
            # set would be a silent no-op that misrepresents where the ids come from.
            body_dict.pop("return_tokens_as_token_ids", None)
            # SGLang's native request extensions for the training metadata.
            body_dict["return_meta_info"] = True
            body_dict["return_prompt_token_ids"] = True
        return body_dict

    async def _attach_token_id_information(
        self, choice_dict: Dict[str, Any], body_dict: Dict[str, Any], client: NeMoGymAsyncOpenAI
    ) -> None:
        """Read the ids/logprobs SGLang already returned, instead of vLLM's `token_id:` parse.

        No `/tokenize` round-trip is needed: `return_prompt_token_ids` makes SGLang report the
        prompt ids it actually tokenized, which is authoritative in a way a local tokenizer
        cannot be.
        """
        # An aborted generation is a truncated fragment, not a completion. Emitting it would
        # put a poisoned rollout into the training batch looking like a normal `stop`.
        if choice_dict.get("finish_reason") == "abort":
            raise RuntimeError(
                f"`{self.config.name}`: SGLang reported finish_reason='abort' (generation was "
                "cancelled server-side, e.g. preemption or shutdown). Refusing to emit a "
                "partial rollout as a completed one."
            )

        prompt_token_ids: Optional[List[int]] = choice_dict.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise RuntimeError(
                f"`{self.config.name}` requested prompt token ids from SGLang "
                "(return_token_id_information=True, so return_prompt_token_ids=True was sent), "
                f"but the response carried none (choice keys={sorted(choice_dict)}). "
                f"transport='chat' requires sglang >= {MIN_SGLANG_VERSION_FOR_CHAT_TRANSPORT}; "
                "use transport='generate' for older builds."
            )
        normalized_prompt_token_ids = _normalize_token_ids(prompt_token_ids, "prompt_token_ids")

        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(choice_dict)

        choice_dict["message"].update(
            dict(
                prompt_token_ids=normalized_prompt_token_ids,
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
            )
        )

        # Clean the duplicated / non-OpenAI fields so the response validates.
        choice_dict.pop("logprobs", None)
        choice_dict.pop("prompt_token_ids", None)
        choice_dict.pop("meta_info", None)

    @classmethod
    def _normalize_sglang_message(cls, message: Any) -> Dict[str, Any]:
        """Snapshot every rendering field while dropping training-only data."""
        if not isinstance(message, dict):
            raise RuntimeError(f"SGLang exact-token sessions require message objects, got {type(message).__name__}")
        normalized = deepcopy(message)
        for field in cls._SGLANG_TRAINING_MESSAGE_FIELDS:
            normalized.pop(field, None)
        if not normalized.get("tool_calls"):
            normalized.pop("tool_calls", None)
        return normalize_tool_call_arguments([normalized])[0]

    @classmethod
    def _normalize_sglang_messages(cls, messages: List[Any]) -> List[Dict[str, Any]]:
        return [cls._normalize_sglang_message(message) for message in messages]

    @staticmethod
    def _normalize_template_ids(encoded: Any) -> List[int]:
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def _render_sglang_token_ids(
        self,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
        *,
        add_generation_prompt: bool,
    ) -> List[int]:
        """Render a complete transcript directly in token space."""
        encoded = self._get_sglang_tokenizer().apply_chat_template(
            self._normalize_sglang_messages(messages),
            tools=tools,
            chat_template=self._get_sglang_chat_template(),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **chat_template_kwargs,
        )
        return self._normalize_template_ids(encoded)

    def _full_sglang_tokenize(
        self,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Render and tokenize a complete prompt on a genuine cache miss."""
        return self._render_sglang_token_ids(
            messages,
            tools,
            chat_template_kwargs,
            add_generation_prompt=True,
        )

    def _sglang_eos_nl(self) -> List[int]:
        """Token ids of the end-of-turn sequence appended when splicing a turn.

        Driven by `sglang_turn_suffix` rather than hardcoded ChatML, because a model whose
        template closes a turn differently would otherwise get a malformed turn boundary
        spliced into every multi-turn prompt.
        """
        if self._sglang_eos_nl_ids is None:
            encoded = self._get_sglang_tokenizer()(
                self.config.sglang_turn_suffix,
                add_special_tokens=False,
            )
            self._sglang_eos_nl_ids = [int(token_id) for token_id in encoded["input_ids"]]
        return self._sglang_eos_nl_ids

    def _sglang_followup_fragment_ids(
        self,
        cached_messages: List[Any],
        cached_sequence: List[int],
        new_messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Prove a continuation boundary in token space and return its suffix."""
        try:
            base_ids = self._render_sglang_token_ids(
                cached_messages,
                tools,
                chat_template_kwargs,
                add_generation_prompt=False,
            )
            continued_messages = list(cached_messages)
            continued_ids = base_ids
            for message in new_messages:
                previous_ids = continued_ids
                continued_messages.append(message)
                continued_ids = self._render_sglang_token_ids(
                    continued_messages,
                    tools,
                    chat_template_kwargs,
                    add_generation_prompt=False,
                )
                if len(continued_ids) <= len(previous_ids) or continued_ids[: len(previous_ids)] != previous_ids:
                    raise RuntimeError(
                        "Unable to prove an exact-token SGLang session splice: "
                        "a continuation message did not strictly extend the rendered transcript"
                    )
            full_ids = self._render_sglang_token_ids(
                continued_messages,
                tools,
                chat_template_kwargs,
                add_generation_prompt=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Unable to prove an exact-token SGLang session splice from the chat template"
            ) from error
        if not base_ids or full_ids[: len(continued_ids)] != continued_ids:
            raise RuntimeError(
                "Unable to prove an exact-token SGLang session splice: the generation-prompt render "
                "does not preserve the tokenized continuation"
            )
        turn_boundary = self._sglang_eos_nl()
        if (
            not turn_boundary
            or base_ids[-len(turn_boundary) :] != turn_boundary
            or cached_sequence[-len(turn_boundary) :] != turn_boundary
        ):
            raise RuntimeError(
                "Unable to prove an exact-token SGLang session splice: the rendered transcript "
                "and cached token sequence do not share the expected terminal turn boundary"
            )
        return full_ids[len(base_ids) :]

    @classmethod
    def _sglang_messages_match(
        cls,
        left: List[Any],
        right: List[Any],
    ) -> bool:
        return cls._normalize_sglang_messages(left) == cls._normalize_sglang_messages(right)

    def _sglang_rendering_sig(
        self,
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[str, str, Optional[str], str]:
        """Identify inputs that affect the cached prompt rendering."""
        return (
            json.dumps(tools, sort_keys=True, default=str),
            json.dumps(chat_template_kwargs, sort_keys=True, default=str),
            self._get_sglang_chat_template(),
            self.config.sglang_turn_suffix,
        )

    @classmethod
    def _normalize_sglang_extra_body(cls, value: Any, source: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"SGLang {source} must decode to an object")

        normalized: Dict[str, Any] = {}
        for key, control_value in deepcopy(value).items():
            canonical_key = cls._SGLANG_EXTRA_BODY_ALIASES.get(key, key)
            if canonical_key in normalized:
                raise ValueError(f"SGLang {source} specifies conflicting aliases for {canonical_key!r}")
            if canonical_key in cls._SGLANG_ADAPTER_OWNED_SAMPLING_FIELDS:
                raise ValueError(f"SGLang {source} cannot override adapter-owned control {canonical_key!r}")
            if canonical_key not in cls._SGLANG_NATIVE_SAMPLING_FIELDS:
                raise ValueError(f"Unsupported SGLang {source} control: {key!r}")
            normalized[canonical_key] = control_value

        n = normalized.get("n")
        if n is not None and n != 1:
            raise NotImplementedError(
                "SGLang /generate exact-token transport supports only n=1 because the "
                "adapter returns one aligned token/logprob sequence"
            )
        return normalized

    @classmethod
    def _parse_sglang_metadata_extra_body(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        raw_extra_body = metadata.get("extra_body")
        if raw_extra_body is None:
            return {}
        try:
            parsed = json.loads(raw_extra_body) if isinstance(raw_extra_body, str) else raw_extra_body
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("SGLang metadata.extra_body must contain a JSON object") from error
        return cls._normalize_sglang_extra_body(parsed, "metadata.extra_body")

    @staticmethod
    def _sglang_json_schema_from_response_format(response_format: Any) -> Optional[str]:
        if hasattr(response_format, "model_dump"):
            response_format = response_format.model_dump(by_alias=True)
        if not isinstance(response_format, dict):
            raise ValueError("SGLang response_format must be an object")

        format_type = response_format.get("type")
        if format_type == "text":
            return None
        if format_type == "json_object":
            return '{"type":"object"}'
        if format_type != "json_schema":
            raise NotImplementedError(f"SGLang /generate does not support response_format type {format_type!r}")

        envelope = response_format.get("json_schema")
        if not isinstance(envelope, dict) or not isinstance(envelope.get("schema"), dict):
            raise ValueError("SGLang json_schema response_format requires json_schema.schema to be an object")
        return json.dumps(envelope["schema"], ensure_ascii=False, separators=(",", ":"))

    def _validate_sglang_request_controls(
        self,
        body_dict: Dict[str, Any],
        tools: Any,
    ) -> None:
        n = body_dict.get("n")
        if n is not None and n != 1:
            raise NotImplementedError(
                "SGLang /generate exact-token transport supports only n=1 because the "
                "adapter returns one aligned token/logprob sequence"
            )

        tool_choice = body_dict.get("tool_choice")
        if tool_choice not in (None, "auto"):
            raise NotImplementedError(
                f"SGLang /generate cannot honor tool_choice={tool_choice!r}; only 'auto' is supported"
            )
        if tools and body_dict.get("parallel_tool_calls") is False:
            raise NotImplementedError(
                "SGLang /generate cannot enforce parallel_tool_calls=False with client-side tool parsing"
            )

        unsupported = {
            key: body_dict.get(key)
            for key in (
                "audio",
                "modalities",
                "prediction",
                "reasoning_effort",
                "service_tier",
                "stream_options",
                "web_search_options",
            )
            if body_dict.get(key) is not None
        }
        if body_dict.get("store") is True:
            unsupported["store"] = True
        if body_dict.get("logprobs") is True:
            unsupported["logprobs"] = True
        if body_dict.get("top_logprobs") is not None:
            unsupported["top_logprobs"] = body_dict["top_logprobs"]
        requested_model = body_dict.get("model")
        if requested_model is not None and requested_model != self.config.model:
            unsupported["model"] = requested_model
        if unsupported:
            raise NotImplementedError(
                "SGLang /generate cannot honor request controls: " + ", ".join(sorted(unsupported))
            )

    def _build_sglang_sampling_params(
        self,
        body_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        remaining_context: int,
    ) -> Dict[str, Any]:
        config_controls = self._normalize_sglang_extra_body(
            self.config.extra_body,
            "config.extra_body",
        )
        metadata_controls = self._parse_sglang_metadata_extra_body(metadata)
        controls = config_controls | metadata_controls

        for key in (
            "temperature",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
        ):
            if key in body_dict:
                if body_dict[key] is None:
                    controls.pop(key, None)
                else:
                    controls[key] = body_dict[key]
        if "seed" in body_dict:
            if body_dict["seed"] is None:
                controls.pop("sampling_seed", None)
            else:
                controls["sampling_seed"] = body_dict["seed"]
        if "response_format" in body_dict:
            if body_dict["response_format"] is None:
                controls.pop("response_format", None)
            else:
                controls["response_format"] = body_dict["response_format"]

        explicit_max_present = "max_completion_tokens" in body_dict or "max_tokens" in body_dict
        if body_dict.get("max_completion_tokens") is not None:
            explicit_max = body_dict["max_completion_tokens"]
        elif "max_tokens" in body_dict:
            explicit_max = body_dict["max_tokens"]
        else:
            explicit_max = None
        requested_max = explicit_max if explicit_max_present else controls.pop("max_new_tokens", None)
        if explicit_max_present and explicit_max is None:
            controls.pop("max_new_tokens", None)
        if requested_max is None:
            max_new_tokens = max(1, remaining_context - 8)
        else:
            if isinstance(requested_max, bool) or not isinstance(requested_max, int) or requested_max < 0:
                raise ValueError(f"SGLang max_new_tokens must be a non-negative integer, got {requested_max!r}")
            max_new_tokens = min(requested_max, remaining_context)

        response_format = controls.pop("response_format", None)
        if response_format is not None:
            if any(controls.get(field) for field in self._SGLANG_GRAMMAR_FIELDS):
                raise ValueError("SGLang response_format conflicts with another constrained-generation control")
            json_schema = self._sglang_json_schema_from_response_format(response_format)
            if json_schema is not None:
                controls["json_schema"] = json_schema

        for field in ("json_schema", "structural_tag"):
            if isinstance(controls.get(field), dict):
                controls[field] = json.dumps(
                    controls[field],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        active_grammars = [field for field in self._SGLANG_GRAMMAR_FIELDS if controls.get(field)]
        if len(active_grammars) > 1:
            raise ValueError(
                "SGLang /generate accepts only one constrained-generation control, got: " + ", ".join(active_grammars)
            )

        min_new_tokens = controls.get("min_new_tokens")
        if min_new_tokens is not None and min_new_tokens > max_new_tokens:
            raise ValueError("SGLang min_new_tokens cannot exceed the context-clamped max_new_tokens")

        sampling_params = {key: value for key, value in controls.items() if value is not None}
        sampling_params["spaces_between_special_tokens"] = False
        sampling_params["max_new_tokens"] = max_new_tokens
        return sampling_params

    def _build_sglang_prompt_ids(
        self,
        request: Request,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], Optional[str]]:
        """Build a prompt, splicing the preceding turn's exact sampled IDs."""
        try:
            session_id = request.session.get(SESSION_ID_KEY)
        except Exception:
            session_id = None
        if session_id is not None:
            state = self._sglang_session_seq.get(session_id)
            rendering_sig = self._sglang_rendering_sig(
                tools,
                chat_template_kwargs,
            )
            if state is not None and state.get("rendering_sig") != rendering_sig:
                raise RuntimeError(
                    "SGLang session tools or chat-template inputs changed after "
                    "sampled tokens were cached. Start a new session instead of "
                    "re-tokenizing the existing trajectory."
                )
            if state is not None:
                cached_messages = state["messages"]
                cached_count = len(cached_messages)
                if len(messages) <= cached_count or not self._sglang_messages_match(
                    messages[:cached_count],
                    cached_messages,
                ):
                    raise RuntimeError(
                        "SGLang session history changed after sampled tokens were cached. "
                        "Start a new session instead of re-tokenizing the trajectory."
                    )
                new_messages = self._normalize_sglang_messages(messages[cached_count:])
                if any(message.get("role") == "assistant" for message in new_messages):
                    raise RuntimeError(
                        "SGLang session history contains an uncached assistant turn; "
                        "start a new session instead of re-tokenizing it"
                    )
                fragment = self._sglang_followup_fragment_ids(
                    cached_messages,
                    state["seq"],
                    new_messages,
                    tools,
                    chat_template_kwargs,
                )
                return list(state["seq"]) + fragment, session_id
        return (
            self._full_sglang_tokenize(
                messages,
                tools,
                chat_template_kwargs,
            ),
            session_id,
        )

    def _update_sglang_session_seq(
        self,
        session_id: Optional[str],
        messages: List[Any],
        prompt_token_ids: List[int],
        generation_token_ids: List[int],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
        assistant_message: Dict[str, Any],
        expected_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache the exact token sequence through the generated assistant turn."""
        if session_id is None:
            return
        eos_newline_ids = self._sglang_eos_nl()
        sequence = list(prompt_token_ids) + list(generation_token_ids)
        max_overlap = min(len(generation_token_ids), len(eos_newline_ids))
        overlap = next(
            (
                overlap_size
                for overlap_size in range(max_overlap, 0, -1)
                if sequence[-overlap_size:] == eos_newline_ids[:overlap_size]
            ),
            0,
        )
        sequence += eos_newline_ids[overlap:]

        if self._sglang_session_seq.get(session_id) is not expected_state:
            raise RuntimeError(
                "SGLang session state changed while a continuation was in flight; "
                "refusing to overwrite the newer exact-token sequence"
            )

        state = {
            "messages": self._normalize_sglang_messages([*messages, assistant_message]),
            "seq": sequence,
            "rendering_sig": self._sglang_rendering_sig(
                tools,
                chat_template_kwargs,
            ),
        }
        self._sglang_session_seq.pop(session_id, None)
        while len(self._sglang_session_seq) >= 8192:
            self._sglang_session_seq.pop(next(iter(self._sglang_session_seq)), None)
        self._sglang_session_seq[session_id] = state

    def _parse_sglang_generation(
        self,
        text: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        """Reconstruct reasoning, visible content, and tool calls from raw text."""
        reasoning_content: Optional[str] = None
        if self.config.uses_reasoning_parser and "</think>" in text:
            reasoning_content, _, remainder = text.partition("</think>")
        else:
            remainder = text

        if self.config.sglang_tool_format == "qwen3_coder":
            tool_calls, content = parse_qwen3_coder_tool_calls(remainder, tools)
            return reasoning_content, content, tool_calls

        tool_calls: List[Dict[str, Any]] = []
        parsed_spans: List[Tuple[int, int]] = []
        for match in self._SGLANG_TOOL_CALL_PATTERN.finditer(remainder):
            block = match.group(1)
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            arguments = parsed.get("arguments", {})
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
                continue
            tool_calls.append(
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
            parsed_spans.append(match.span())

        content_parts: List[str] = []
        cursor = 0
        for start, end in parsed_spans:
            content_parts.append(remainder[cursor:start])
            cursor = end
        content_parts.append(remainder[cursor:])
        content = "".join(content_parts).strip()
        return reasoning_content, content, tool_calls

    @staticmethod
    def _is_sglang_context_length_error(error: ClientResponseError) -> bool:
        if error.status != 400:
            return False
        raw_body = getattr(error, "response_content", None)
        try:
            if isinstance(raw_body, bytes):
                raw_body = raw_body.decode()
            if not isinstance(raw_body, str):
                return False
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        envelope = payload.get("error")
        if not isinstance(envelope, dict) or not isinstance(envelope.get("message"), str):
            return False
        message = envelope["message"]
        return bool(
            re.fullmatch(
                r"The input \(\d+ tokens\) is longer than the model's context length \(\d+ tokens\)\.",
                message,
            )
            or re.match(
                r"^Requested token count exceeds the model's maximum context length of \d+ tokens\.",
                message,
            )
        )

    def _sglang_length_finish(
        self,
        prompt_token_ids: List[int],
    ) -> NeMoGymChatCompletion:
        """Terminate an over-context turn without truncating its prompt."""
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }
        if self.config.return_token_id_information:
            message.update(
                {
                    "prompt_token_ids": list(prompt_token_ids),
                    "generation_token_ids": [],
                    "generation_log_probs": [],
                }
            )
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"chtcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time()),
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": message,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt_token_ids),
                },
            }
        )

    async def _sglang_chat_completion(
        self,
        request: Request,
        body_dict: Dict[str, Any],
    ) -> NeMoGymChatCompletion:
        """Generate exact training tokens through SGLang's native endpoint."""
        client = self._resolve_client(request)

        messages = body_dict["messages"]
        if self.config.replace_developer_role_with_system:
            for message in messages:
                if message.get("role") == "developer":
                    message["role"] = "system"
        tools = body_dict.get("tools")
        self._validate_sglang_request_controls(body_dict, tools)

        chat_template_kwargs: Dict[str, Any] = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)
        metadata = body_dict.get("metadata") or {}
        chat_template_kwargs.update(
            json.loads(metadata.get("chat_template_kwargs", "{}")),
        )

        tokenizer = self._get_sglang_tokenizer()
        prompt_token_ids, session_id = self._build_sglang_prompt_ids(
            request,
            messages,
            tools,
            chat_template_kwargs,
        )
        expected_state = self._sglang_session_seq.get(session_id) if session_id is not None else None

        remaining_context = self.config.context_length - len(prompt_token_ids)
        if remaining_context <= 0:
            return self._sglang_length_finish(prompt_token_ids)

        sampling_params = self._build_sglang_sampling_params(
            body_dict,
            metadata,
            remaining_context,
        )

        try:
            result = await client.create_generate(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                return_logprob=True,
            )
        except ClientResponseError as error:
            if self._is_sglang_context_length_error(error):
                return self._sglang_length_finish(prompt_token_ids)
            raise

        raw_meta_info = result.get("meta_info")
        meta_info = raw_meta_info if isinstance(raw_meta_info, dict) else {}
        finish = meta_info.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        if finish == "abort":
            raise RuntimeError(
                f"`{self.config.name}`: SGLang reported finish_reason='abort' (generation was "
                "cancelled server-side). Refusing to emit or cache a partial rollout."
            )

        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(
            result,
        )

        generated_text = tokenizer.decode(
            generation_token_ids,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        stripped = True
        while stripped:
            stripped = False
            generated_text = generated_text.rstrip("\n")
            for eos_marker in self.config.sglang_eos_markers:
                if generated_text.endswith(eos_marker):
                    generated_text = generated_text[: -len(eos_marker)]
                    stripped = True
        reasoning_content, content, tool_calls = self._parse_sglang_generation(
            generated_text,
            tools=tools,
        )

        if finish == "length":
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        if self.config.uses_reasoning_parser and reasoning_content:
            content = self._converter._wrap_reasoning_in_think_tags([reasoning_content]) + (content or "")

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls or None,
        }
        if self.config.return_token_id_information:
            message.update(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "generation_token_ids": generation_token_ids,
                    "generation_log_probs": generation_log_probs,
                }
            )

        response = NeMoGymChatCompletion.model_validate(
            {
                "id": f"chtcmpl-{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time()),
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": message,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(generation_token_ids),
                    "total_tokens": len(prompt_token_ids) + len(generation_token_ids),
                },
            }
        )
        self._update_sglang_session_seq(
            session_id,
            messages,
            prompt_token_ids,
            generation_token_ids,
            tools,
            chat_template_kwargs,
            assistant_message=message,
            expected_state=expected_state,
        )
        return response


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SGLangModel.run_webserver()  # noqa: F401
