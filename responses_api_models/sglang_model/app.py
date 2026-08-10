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
    extract_generated_tokens_and_logprobs,
    unsupported_sampling_params,
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
        return self


class SGLangModel(VLLMModel):
    """Responses-API adapter that preserves exact sampled token IDs."""

    config: SGLangModelConfig

    _SGLANG_TOOL_CALL_PATTERN: ClassVar = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )
    _SGLANG_ARGS_PATTERN: ClassVar = re.compile(
        r'"arguments"\s*:\s*(.*)\}\s*$',
        re.DOTALL,
    )

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
        if self.config.transport == "chat":
            # Inherit the whole vLLM path; only token extraction differs (see
            # `_attach_token_id_information`).
            return await super().chat_completions(request, body)
        # `/generate` needs a pre-tokenized prompt, so it deliberately bypasses the
        # vLLM-specific request preprocessing and renders locally instead.
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

        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(choice_dict)

        prompt_token_ids: Optional[List[int]] = choice_dict.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise RuntimeError(
                f"`{self.config.name}` requested prompt token ids from SGLang "
                "(return_token_id_information=True, so return_prompt_token_ids=True was sent), "
                f"but the response carried none (choice keys={sorted(choice_dict)}). "
                f"transport='chat' requires sglang >= {MIN_SGLANG_VERSION_FOR_CHAT_TRANSPORT}; "
                "use transport='generate' for older builds."
            )

        choice_dict["message"].update(
            dict(
                prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
                generation_token_ids=generation_token_ids,
                generation_log_probs=generation_log_probs,
            )
        )

        # Clean the duplicated / non-OpenAI fields so the response validates.
        choice_dict.pop("logprobs", None)
        choice_dict.pop("prompt_token_ids", None)
        choice_dict.pop("meta_info", None)

    def _full_sglang_tokenize(
        self,
        messages: List[Any],
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Render and tokenize a complete prompt on a cache miss."""
        encoded = self._get_sglang_tokenizer().apply_chat_template(
            normalize_tool_call_arguments(messages),
            tools=tools,
            chat_template=self._get_sglang_chat_template(),
            add_generation_prompt=True,
            tokenize=True,
            **chat_template_kwargs,
        )
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

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
        new_messages: List[Any],
        chat_template_kwargs: Dict[str, Any],
    ) -> Optional[List[int]]:
        """Render the new messages and next assistant header as a token fragment.

        The fragment is derived by differencing two template renders against an
        anchor assistant turn. Returning ``None`` asks the caller to fall back
        to a complete render when the template is not splice-friendly.
        """
        tokenizer = self._get_sglang_tokenizer()
        chat_template = self._get_sglang_chat_template()
        anchor = [{"role": "assistant", "content": "X"}]
        try:
            full = tokenizer.apply_chat_template(
                anchor + list(new_messages),
                tools=None,
                chat_template=chat_template,
                add_generation_prompt=True,
                tokenize=False,
                **chat_template_kwargs,
            )
            base = tokenizer.apply_chat_template(
                anchor,
                tools=None,
                chat_template=chat_template,
                add_generation_prompt=False,
                tokenize=False,
                **chat_template_kwargs,
            )
        except Exception:
            return None
        if not isinstance(full, str) or not isinstance(base, str) or not full.startswith(base):
            return None
        encoded = tokenizer(full[len(base) :], add_special_tokens=False)
        return [int(token_id) for token_id in encoded["input_ids"]]

    @staticmethod
    def _sglang_msg_sig(message: Dict[str, Any]) -> Tuple[Any, str, str]:
        return (
            message.get("role"),
            json.dumps(message.get("content"), sort_keys=True, default=str),
            json.dumps(message.get("tool_calls"), sort_keys=True, default=str),
        )

    @classmethod
    def _sglang_messages_match(
        cls,
        left: List[Any],
        right: List[Any],
    ) -> bool:
        return len(left) == len(right) and all(
            cls._sglang_msg_sig(left_message) == cls._sglang_msg_sig(right_message)
            for left_message, right_message in zip(left, right)
        )

    def _sglang_rendering_sig(
        self,
        tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[str, str, Optional[str]]:
        """Identify inputs that affect the cached prompt rendering."""
        return (
            json.dumps(tools, sort_keys=True, default=str),
            json.dumps(chat_template_kwargs, sort_keys=True, default=str),
            self._get_sglang_chat_template(),
        )

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
                previous_messages = state["messages"]
                previous_count = len(previous_messages)
                if (
                    len(messages) > previous_count
                    and messages[previous_count].get("role") == "assistant"
                    and all(message.get("role") != "assistant" for message in messages[previous_count + 1 :])
                    and self._sglang_messages_match(
                        messages[:previous_count],
                        previous_messages,
                    )
                ):
                    fragment = self._sglang_followup_fragment_ids(
                        messages[previous_count + 1 :],
                        chat_template_kwargs,
                    )
                    if fragment is not None:
                        return state["seq"] + fragment, session_id
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
    ) -> None:
        """Cache the exact token sequence through the generated assistant turn."""
        if session_id is None:
            return
        eos_newline_ids = self._sglang_eos_nl()
        sequence = list(prompt_token_ids) + list(generation_token_ids)
        max_overlap = min(len(sequence), len(eos_newline_ids))
        overlap = next(
            (
                overlap_size
                for overlap_size in range(max_overlap, 0, -1)
                if sequence[-overlap_size:] == eos_newline_ids[:overlap_size]
            ),
            0,
        )
        sequence += eos_newline_ids[overlap:]

        self._sglang_session_seq.pop(session_id, None)
        while len(self._sglang_session_seq) >= 8192:
            self._sglang_session_seq.pop(next(iter(self._sglang_session_seq)), None)
        self._sglang_session_seq[session_id] = {
            "messages": list(messages),
            "seq": sequence,
            "rendering_sig": self._sglang_rendering_sig(
                tools,
                chat_template_kwargs,
            ),
        }

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
        for match in self._SGLANG_TOOL_CALL_PATTERN.finditer(remainder):
            block = match.group(1)
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            arguments_match = self._SGLANG_ARGS_PATTERN.search(block)
            arguments = (
                arguments_match.group(1).strip()
                if arguments_match is not None
                else json.dumps(parsed.get("arguments", {}))
            )
            tool_calls.append(
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": parsed.get("name"),
                        "arguments": arguments,
                    },
                }
            )

        content = self._SGLANG_TOOL_CALL_PATTERN.sub("", remainder).strip()
        return reasoning_content, content, tool_calls

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

        remaining_context = self.config.context_length - len(prompt_token_ids)
        if remaining_context <= 0:
            return self._sglang_length_finish(prompt_token_ids)

        sampling_params: Dict[str, Any] = {"spaces_between_special_tokens": False}
        max_new_tokens = body_dict.get("max_completion_tokens") or body_dict.get("max_tokens") or None
        if max_new_tokens is None:
            max_new_tokens = remaining_context - 8
            max_new_tokens = max(1, max_new_tokens)
        else:
            max_new_tokens = min(max_new_tokens, remaining_context)
        sampling_params["max_new_tokens"] = max_new_tokens
        for key in ("temperature", "top_p", "top_k", "stop", "frequency_penalty", "repetition_penalty", "min_p"):
            if body_dict.get(key) is not None:
                sampling_params[key] = body_dict[key]
        ignored = unsupported_sampling_params(body_dict)
        if ignored:
            print(
                f"[sglang_model] transport='generate' cannot honor {ignored}; the realized "
                "sampling distribution will differ from the configured recipe. Use "
                "transport='chat' (sglang >= 0.5.13) for full parameter support.",
                flush=True,
            )

        try:
            result = await client.create_generate(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                return_logprob=True,
            )
        except ClientResponseError as error:
            try:
                error_body = error.response_content.decode()
            except Exception:
                error_body = str(error)
            if any(
                fragment in error_body
                for fragment in (
                    "context length",
                    "longer than",
                    "max_total",
                    "is longer",
                )
            ):
                return self._sglang_length_finish(prompt_token_ids)
            raise

        meta_info = result.get("meta_info") or {}
        generation_token_ids, generation_log_probs = extract_generated_tokens_and_logprobs(
            result,
        )
        self._update_sglang_session_seq(
            session_id,
            messages,
            prompt_token_ids,
            generation_token_ids,
            tools,
            chat_template_kwargs,
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

        finish = meta_info.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type")
        if finish == "abort":
            # A truncated fragment, not a completion. Reporting it as `stop` would put a
            # poisoned rollout into the training batch looking like a normal turn.
            raise RuntimeError(
                f"`{self.config.name}`: SGLang reported finish_reason='abort' (generation was "
                "cancelled server-side, e.g. preemption or shutdown). Refusing to emit a "
                "partial rollout as a completed one."
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

        return NeMoGymChatCompletion.model_validate(
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


if __name__ == "__main__":
    SGLangModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SGLangModel.run_webserver()  # noqa: F401
