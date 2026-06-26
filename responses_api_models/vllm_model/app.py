# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from copy import deepcopy
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4


async def _sglang_teacher_force(base_url):
    """In-proxy SGLang teacher-force via /generate + logprob_start_len -> input_token_logprobs
    (raw per-token logprobs of GIVEN output tokens) with top_logprobs_num for KL."""
    import json, httpx
    P = "/tmp/swe2_parity"
    dbg = P + "/sglang_tf.log"
    b = (base_url or "").rstrip("/")
    gen_url = (b[:-3].rstrip("/") if b.endswith("/v1") else b) + "/generate"
    inp, out = P + "/rollouts_filtered16.jsonl", P + "/forced_sglang.jsonl"
    def log(m):
        try:
            with open(dbg, "a") as f: f.write(str(m) + "\n")
        except Exception: pass
    log("START gen_url=%s" % gen_url)
    try:
        recs = [json.loads(l) for l in open(inp) if l.strip()][:100]
    except Exception as e:
        log("READ_FAIL %s" % e); return
    n = 0
    async with httpx.AsyncClient(timeout=600) as cx:
        with open(out, "w") as fo:
            for rec in recs:
                p = rec["prompt_token_ids"]; g = rec["generation_token_ids"]
                if not g: continue
                full = [int(x) for x in p] + [int(x) for x in g]; L = len(p)
                start = max(0, L - 1)
                payload = {"input_ids": full, "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": start, "top_logprobs_num": 20}
                try:
                    r = await cx.post(gen_url, json=payload)
                    meta = (r.json().get("meta_info") or {})
                    itl = meta.get("input_token_logprobs") or []
                    itop = meta.get("input_top_logprobs") or []
                except Exception as e:
                    if n == 0: log("POST_FAIL %s" % e)
                    fo.write(json.dumps({"backend": "sglang", "error": str(e)[:200], "n_gen": len(g)}) + "\n"); n += 1; continue
                by_pos = {}
                for k, entry in enumerate(itl):
                    pos = start + k
                    top = []
                    if k < len(itop) and itop[k]:
                        top = [[e[0], int(e[1])] for e in itop[k]]
                    by_pos[pos] = {"token_id": int(entry[1]), "logprob": entry[0], "top": top}
                forced = []
                for j, tid in enumerate(g):
                    pos = L + j; e = by_pos.get(pos)
                    if e is None or e["token_id"] != int(tid) or e["logprob"] is None:
                        forced.append({"pos": j, "token_id": int(tid), "logprob": None, "top": [], "align_ok": False}); continue
                    forced.append({"pos": j, "token_id": int(tid), "logprob": e["logprob"], "top": e["top"], "align_ok": True})
                fo.write(json.dumps({"backend": "sglang", "n_prompt": L, "n_gen": len(g),
                                     "forced": forced, "sampled_log_probs": rec.get("generation_log_probs")}) + "\n")
                fo.flush(); n += 1; log("rec %d/%d n_gen=%d" % (n, len(recs), len(g)))
    log("DONE %d" % n)
    try: open(P + "/SGLANG_TF_DONE", "w").write("done")
    except Exception: pass


async def _sglang_greedy_redecode(base_url):
    """Background: greedy-decode (temp=0) the captured vLLM prompts on the LOCAL SGLang
    server via /generate, capturing SGLang's greedy tokens + logprobs. Carries vLLM's
    greedy tokens/logprobs through for paired agreement comparison."""
    import json, httpx
    P = "/tmp/swe2_parity"
    dbg = P + "/sglang_greedy.log"
    b = (base_url or "").rstrip("/")
    gen_url = (b[:-3].rstrip("/") if b.endswith("/v1") else b) + "/generate"
    inp, out = P + "/vllm_greedy_filtered.jsonl", P + "/greedy_sglang.jsonl"
    def log(m):
        try:
            with open(dbg, "a") as f: f.write(str(m) + "\n")
        except Exception: pass
    log("START gen_url=%s" % gen_url)
    try:
        recs = [json.loads(l) for l in open(inp) if l.strip()][:100]
    except Exception as e:
        log("READ_FAIL %s" % e); return
    n = 0
    async with httpx.AsyncClient(timeout=600) as cx:
        with open(out, "w") as fo:
            for rec in recs:
                p = rec["prompt_token_ids"]; vg = rec["generation_token_ids"]; vl = rec.get("generation_log_probs")
                mnt = min(len(vg) + 4, 4000)
                payload = {"input_ids": [int(x) for x in p],
                           "sampling_params": {"temperature": 0, "max_new_tokens": mnt},
                           "return_logprob": True}
                try:
                    r = await cx.post(gen_url, json=payload)
                    if r.status_code != 200:
                        if n == 0: log("HTTP %d %s" % (r.status_code, r.text[:200]))
                        fo.write(json.dumps({"error": "http%d" % r.status_code}) + "\n"); n += 1; continue
                    otl = (r.json().get("meta_info") or {}).get("output_token_logprobs") or []
                except Exception as e:
                    if n == 0: log("POST_FAIL %s" % e)
                    fo.write(json.dumps({"error": str(e)[:200]}) + "\n"); n += 1; continue
                sg_ids = [int(t[1]) for t in otl]; sg_lp = [float(t[0]) for t in otl]
                fo.write(json.dumps({"n_prompt": len(p), "sglang_gen_ids": sg_ids, "sglang_log_probs": sg_lp,
                                     "vllm_gen_ids": [int(x) for x in vg], "vllm_log_probs": vl}) + "\n")
                fo.flush(); n += 1
                log("rec %d/%d sg_gen=%d vllm_gen=%d" % (n, len(recs), len(sg_ids), len(vg)))
    log("DONE %d" % n)
    try:
        open(P + "/SGLANG_GREEDY_DONE", "w").write("done")
    except Exception: pass


async def _vllm_probe_routes(base_url, model):
    import json, httpx
    P = "/tmp/swe2_parity"
    dbg = P + "/vllm_probe.log"
    b = (base_url or "").rstrip("/")
    root = b[:-3].rstrip("/") if b.endswith("/v1") else b
    def log(m):
        try:
            with open(dbg, "a") as f: f.write(str(m) + "\n")
        except Exception: pass
    log("PROBE base_url=%s root=%s model=%s" % (base_url, root, model))
    toks = [785, 374, 264, 1273]
    async with httpx.AsyncClient(timeout=40) as cx:
        try:
            r = await cx.get(root + "/openapi.json")
            if r.status_code == 200:
                paths = sorted((r.json().get("paths") or {}).keys())
                log("ROUTES: " + json.dumps(paths))
            else:
                log("GET /openapi.json -> %d" % r.status_code)
        except Exception as e:
            log("GET /openapi.json FAIL %s" % e)
        tests = [
            (root + "/v1/completions", {"model": model, "prompt": toks, "echo": True, "max_tokens": 1, "prompt_logprobs": 1}),
            (root + "/completions", {"model": model, "prompt": toks, "echo": True, "max_tokens": 1, "prompt_logprobs": 1}),
            (root + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "prompt_logprobs": 1, "logprobs": True, "top_logprobs": 1}),
            (root + "/score", {"model": model, "text_1": "a", "text_2": "b"}),
            (root + "/v1/score", {"model": model, "text_1": "a", "text_2": "b"}),
            (root + "/pooling", {"model": model, "input": toks}),
        ]
        for u, pl in tests:
            try:
                r = await cx.post(u, json=pl)
                body = r.text[:280].replace(chr(10), " ")
                has_plp = ("prompt_logprobs" in r.text)
                log("POST %s -> %d has_prompt_logprobs=%s | %s" % (u, r.status_code, has_plp, body))
            except Exception as e:
                log("POST %s FAIL %s" % (u, e))
    log("PROBE_DONE")
    try:
        open(P + "/VLLM_PROBE_DONE", "w").write("done")
    except Exception: pass


async def _vllm_parity_teacher_force(base_url, model):
    """Background: teacher-force vLLM on captured tokens via /v1/completions (echo +
    prompt_logprobs), using the proxy's own reachability to the vLLM server."""
    import os, json, httpx
    P = "/tmp/swe2_parity"
    inp, out, dbg = P + "/rollouts_filtered.jsonl", P + "/forced_vllm.jsonl", P + "/vllm_tf_debug.log"
    b = (base_url or "").rstrip("/")
    url = b + "/completions" if b.endswith("/v1") else b + "/v1/completions"
    def log(m):
        try:
            with open(dbg, "a") as f: f.write(str(m) + "\n")
        except Exception: pass
    log("START base_url=%s url=%s model=%s" % (base_url, url, model))
    try:
        recs = [json.loads(l) for l in open(inp) if l.strip()][:100]
    except Exception as e:
        log("READ_FAIL %s" % e); return
    n = 0
    async with httpx.AsyncClient(timeout=600) as cx:
        with open(out, "w") as fo:
            for rec in recs:
                p, g = rec.get("prompt_token_ids"), rec.get("generation_token_ids")
                if not g: continue
                full = [int(x) for x in p] + [int(x) for x in g]; L = len(p)
                payload = {"model": model, "prompt": full, "echo": True,
                           "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20}
                try:
                    r = await cx.post(url, json=payload)
                    if r.status_code != 200:
                        if n == 0: log("HTTP %d: %s" % (r.status_code, r.text[:300]))
                        fo.write(json.dumps({"backend": "vllm", "error": "http%d" % r.status_code, "n_gen": len(g)}) + "\n"); n += 1; continue
                    plp = (r.json()["choices"][0].get("prompt_logprobs")) or []
                except Exception as e:
                    if n == 0: log("POST_FAIL %s" % e)
                    fo.write(json.dumps({"backend": "vllm", "error": str(e)[:200], "n_gen": len(g)}) + "\n"); n += 1; continue
                forced = []
                for j, tid in enumerate(g):
                    tid = int(tid); pos = L + j
                    entry = plp[pos] if pos < len(plp) else None
                    if not entry:
                        forced.append({"pos": j, "token_id": tid, "logprob": None, "top": [], "align_ok": False}); continue
                    e2 = entry.get(str(tid))
                    top = []
                    for k, v in entry.items():
                        try: top.append([float(v["logprob"]), int(k)])
                        except Exception: pass
                    forced.append({"pos": j, "token_id": tid,
                                   "logprob": (float(e2["logprob"]) if e2 else None),
                                   "top": top, "align_ok": e2 is not None})
                fo.write(json.dumps({"backend": "vllm", "n_prompt": L, "n_gen": len(g),
                                     "forced": forced, "sampled_log_probs": rec.get("generation_log_probs")}) + "\n")
                fo.flush(); n += 1
                log("rec %d/%d n_gen=%d" % (n, len(recs), len(g)))
    log("DONE wrote %d" % n)
    try: open(P + "/VLLM_TF_DONE", "w").write("done")
    except Exception: pass


def _maybe_launch_vllm_parity_tf(config):
    import os, asyncio
    P = "/tmp/swe2_parity"
    _bu0 = config.base_url
    _bu = _bu0[0] if isinstance(_bu0, (list, tuple)) else _bu0
    if os.path.exists(P + "/SGLANG_TF_TRIGGER"):
        try:
            _fd = os.open(P + "/SGLANG_TF.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(_fd)
        except Exception:
            return
        try:
            asyncio.create_task(_sglang_teacher_force(_bu))
        except Exception:
            pass
        return
    if os.path.exists(P + "/SGLANG_GREEDY_TRIGGER"):
        try:
            _fd = os.open(P + "/SGLANG_GREEDY.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(_fd)
        except Exception:
            return
        try:
            asyncio.create_task(_sglang_greedy_redecode(_bu))
        except Exception:
            pass
        return
    if os.path.exists(P + "/VLLM_PROBE_TRIGGER"):
        try:
            _fd = os.open(P + "/VLLM_PROBE.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(_fd)
        except Exception:
            return
        try:
            asyncio.create_task(_vllm_probe_routes(_bu, config.model))
        except Exception:
            pass
        return
    if not os.path.exists(P + "/VLLM_TF_TRIGGER"):
        return
    try:
        fd = os.open(P + "/VLLM_TF.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
    except Exception:
        return
    bu = config.base_url
    base_url = bu[0] if isinstance(bu, (list, tuple)) else bu
    try:
        asyncio.create_task(_vllm_parity_teacher_force(base_url, config.model))
    except Exception:
        pass


def _logprob_parity_dump(engine, prompt_token_ids, generation_token_ids, generation_log_probs):
    """Append one rollout turn's exact token ids + sampled logprobs as a JSON line.
    Trigger (zero cost when off; best-effort, never raises into the hot path):
      1. env LOGPROB_PARITY_DUMP=<outpath>  (works when env reaches this process), else
      2. a sentinel file on shared lustre whose contents are the <outpath>:
         /tmp/swe2_parity/DUMP_PATH
    The sentinel path works regardless of Ray actor placement / node (lustre is shared).
    Each process appends to <outpath>.<pid> to avoid interleaving."""
    import os
    path = os.environ.get("LOGPROB_PARITY_DUMP")
    if not path:
        sentinel = "/tmp/swe2_parity/DUMP_PATH"
        try:
            if os.path.exists(sentinel):
                with open(sentinel) as _s:
                    path = _s.read().strip()
        except Exception:
            path = None
    if not path or not generation_token_ids:
        return
    try:
        rec = {
            "engine": engine,
            "prompt_token_ids": [int(t) for t in prompt_token_ids],
            "generation_token_ids": [int(t) for t in generation_token_ids],
            "generation_log_probs": [float(x) for x in generation_log_probs],
        }
        with open("%s.%d" % (path, os.getpid()), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request
from pydantic import BaseModel, Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionDeveloperMessageParam,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageParam,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionToolParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionDefinition,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
    TokenIDLogProbMixin,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_worker


class VLLMModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool

    uses_reasoning_parser: bool
    replace_developer_role_with_system: bool = False

    # Whether or not the model can generate a reasoning output, and called again to produce additional reasoning output.
    sequential_reasoning_allowed: bool = True

    # As of Feb 2026, we default this to False since majority of open source models aren't responses native with the exception of GPT-OSS
    is_responses_native: bool = False

    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Corresponds to the extra_body of OpenAI Client.
    extra_body: Optional[Dict[str, Any]] = None

    # Generation engine. "vllm" (default) keeps the original OpenAI /v1/chat/completions
    # marshaling path byte-for-byte. "sglang" switches to SGLang's native /generate
    # endpoint (see VLLMModel._sglang_chat_completion): on the pinned SGLang v0.5.10 the
    # chat endpoint cannot return the exact sampled integer token ids (decoded-string
    # logprobs, no return_tokens_as_token_ids) and /tokenize only accepts a raw prompt
    # string, so the proxy tokenizes locally and reads token ids from /generate's
    # meta_info.output_token_logprobs instead.
    engine: Literal["vllm", "sglang"] = "vllm"

    # Path to the Jinja chat template the SGLang server was launched with (--chat-template).
    # Used (engine == "sglang") so the proxy renders prompts with the same template the
    # model was served/trained with. If None, the model tokenizer's built-in template is used.
    sglang_chat_template_path: Optional[str] = None

    # Max sequence length of the SGLang server (engine == "sglang"). When a request does not
    # specify a positive max_tokens (the SWE agent sends max_output_tokens=0 = "unlimited"),
    # the proxy fills the remaining context as max_new_tokens. Without this, SGLang /generate
    # falls back to its default max_new_tokens=128, truncating reasoning before </think> and
    # breaking multi-turn contiguity. Mirrors the recipe's sglang_cfg.max_new_tokens.
    sglang_max_total_sequence_length: Optional[int] = None

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


class VLLMModel(SimpleResponsesAPIModel):
    config: VLLMModelConfig

    def get_converter(self) -> "VLLMConverter":
        """Return the converter used for Responses API <-> Chat Completions mapping.

        Override in subclasses (e.g. GenRMModel) to use a specialized converter.
        """
        return VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
        )

    def model_post_init(self, context):
        self._post_init()
        return super().model_post_init(context)

    def _post_init(self) -> None:
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.api_key,
            )
            for base_url in self.config.base_url
        ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()

        self._converter = self.get_converter()

        # Lazily-initialised state for the SGLang engine path (see _sglang_chat_completion).
        self._sglang_tokenizer: Any = None
        self._sglang_chat_template: Optional[str] = None
        # Contiguity fix: per-session running token sequence. Each multi-turn rollout's prompt
        # is built by splicing the prior assistant turn's EXACT sampled generation_token_ids
        # (never re-tokenizing them), so nemo_gym.py's `seen == prompt[:len(seen)]` holds by
        # construction. Re-tokenizing prior turns broke this two ways: proxy parse drift
        # (dropped multi-line tool calls, mangled </think>) and BPE retokenization (identical
        # text, different token split). Keyed by SESSION_ID_KEY; cache-miss -> full tokenize.
        self._sglang_session_seq: Dict[str, Dict[str, Any]] = dict()
        self._sglang_eos_nl_ids: Optional[List[int]] = None

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.is_responses_native:
            return await self._responses_native(request, body)

        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        choice = chat_completion_response.choices[0]

        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion_response.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion_response.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion_response.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion_response.usage.prompt_tokens
                + chat_completion_response.usage.completion_tokens,
            )

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if "tool_choice" in body else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details={"reason": "max_output_tokens"} if choice.finish_reason == "length" else None,
            usage=usage,
        )

    async def _responses_native(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """
        The following config parameters are effectively no-ops with Responses native models:
        - uses_reasoning_parser: bool (Not applicable)
        """
        # The following parameters could be supported, but have not been supported yet for Responses-native models:
        if self.config.return_token_id_information:
            raise NotImplementedError
        if self.config.replace_developer_role_with_system:
            raise NotImplementedError
        if not self.config.sequential_reasoning_allowed:
            raise NotImplementedError

        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model
        if self.config.chat_template_kwargs:
            body_dict["chat_template_kwargs"] = deepcopy(self.config.chat_template_kwargs)
        if self.config.extra_body:
            body_dict = self.config.extra_body | body_dict

        client = self._resolve_client(request)
        response_dict = await client.create_response(**body_dict)

        return NeMoGymResponse.model_validate(response_dict)

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the body dict before issuing a chat completion request.

        Subclasses can override this to apply model-specific transformations
        (e.g. role remapping, extra sampling params).  The base implementation
        handles the features driven by ``VLLMModelConfig``.

        Args:
            request: The originating FastAPI request (available for session /
                client resolution if needed by subclasses).
            body_dict: Mutable dict produced by ``body.model_dump(exclude_unset=True)``.

        Returns:
            The (possibly mutated) ``body_dict`` that will be forwarded to
            ``client.create_chat_completion``.
        """
        if self.config.replace_developer_role_with_system:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"

        body_dict["model"] = self.config.model

        chat_template_kwargs = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)

        metadata = body_dict.get("metadata", dict())

        # Merge global config chat_template_kwargs with per-request overrides in metadata (e.g. per-sample reasoning on/off)
        metadata_chat_template_kwargs_str = metadata.get("chat_template_kwargs", "{}")
        chat_template_kwargs.update(json.loads(metadata_chat_template_kwargs_str))

        if chat_template_kwargs:
            body_dict["chat_template_kwargs"] = chat_template_kwargs

        # Merge global config extra_body with per-request overrides from metadata
        extra_body = {}
        if self.config.extra_body:
            extra_body = deepcopy(self.config.extra_body)

        metadata_extra_body_str = metadata.get("extra_body", "{}")
        extra_body.update(json.loads(metadata_extra_body_str))

        if self.config.return_token_id_information:
            body_dict |= dict(
                logprobs=True,
                # Typically passed via OpenAI client extra_body.
                return_tokens_as_token_ids=True,
                # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                # For prompt and generation token IDs
                # return_token_ids=True,
                # For prompt token IDs
                # prompt_logprobs=0,
            )

        if self.config.uses_reasoning_parser:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue

                content = message_dict["content"]
                if isinstance(content, str):
                    reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(content)
                    message_dict["content"] = remaining_content
                    if reasoning_matches:
                        message_dict["reasoning_content"] = reasoning_matches[0]

                        # TODO when NeMo RL migrates to vLLM>=0.16.0, remove the reasoning_content support above.
                        # Starting with vLLM 0.16.0, the `reasoning_content` field has been deprecated in favor of just `reasoning`
                        message_dict["reasoning"] = reasoning_matches[0]
                elif isinstance(content, list):
                    reasoning_content = None
                    for content_item_dict in content:
                        reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(
                            content_item_dict["text"]
                        )
                        assert reasoning_content is None or not reasoning_matches, (
                            f"Found multiple reasoning matches in a single assistant message content item list!\nMessage: {message_dict}"
                        )

                        # Even though we set the reasoning content already here, we still loop through all the content item dicts for the assert above.
                        content_item_dict["text"] = remaining_content
                        if reasoning_matches:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                            # See the TODO wrt reasoning_content above
                            message_dict["reasoning"] = reasoning_matches[0]
                elif not content:
                    # No content or content None is a no-op
                    pass
                else:
                    raise NotImplementedError

        if extra_body:
            body_dict = extra_body | body_dict

        return body_dict

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)

        # SGLang engine path: handled entirely by _sglang_chat_completion, which renders the
        # prompt locally (keeping <think> embedded in assistant content, as the SWE chat
        # template expects) and generates via /generate. Dispatched BEFORE the vLLM-specific
        # _preprocess (which would split reasoning out of assistant content for the chat API).
        if self.config.engine == "sglang":
            return await self._sglang_chat_completion(request, body_dict)

        body_dict = self._preprocess_chat_completion_create_params(request, body_dict)

        client = self._resolve_client(request)

        _maybe_launch_vllm_parity_tf(self.config)

        if not self.config.sequential_reasoning_allowed:
            last_message = body_dict["messages"][-1]
            if last_message["role"] == "assistant" and not (last_message["content"] or last_message.get("tool_calls")):
                return self._create_empty_chat_completion()

        try:
            chat_completion_dict = await client.create_chat_completion(**body_dict)
        except ClientResponseError as e:
            """
            Example messages for out of context length:

            1. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L914
            ```json
            {"object":"error","message":"This model\'s maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages, Please reduce the length of the messages. None","type":"BadRequestError","param":null,"code":400}
            ```
            2. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L940
            3. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L948
            4. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/sampling_params.py#L463
            """
            result_content_str = e.response_content.decode()

            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                return NeMoGymChatCompletion(
                    id="chtcmpl-123",
                    object="chat.completion",
                    created=int(time()),
                    model=self.config.model,
                    choices=[
                        NeMoGymChoice(
                            index=0,
                            finish_reason="stop",
                            message=NeMoGymChatCompletionMessage(
                                role="assistant",
                                content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                )
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
        if self.config.uses_reasoning_parser:
            # See the TODO wrt reasoning_content above
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                # See the TODO wrt reasoning_content above
                choice_dict["message"].pop("reasoning", None)

                # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"]["content"] or "")
        else:
            # See the TODO wrt reasoning_content above
            assert not (choice_dict["message"].get("reasoning_content") or choice_dict["message"].get("reasoning")), (
                f"NeMo Gym server `{self.config.name}` config has explicitly been set to not use a reasoning parser i.e. `uses_reasoning_parser: false`. Please do not use a reasoning parser in your vLLM endpoint, or fix the `{self.config.name}` server config!"
            )

        if self.config.return_token_id_information:
            log_probs = choice_dict["logprobs"]["content"]
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            """
            START TODO remove this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            """
            # Looks like `"token_id:151667"`
            generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            # The tokenize endpoint doesn't accept any sampling parameters
            # The only relevant params are model, messages, and tools.
            #
            # IMPORTANT: pass through chat-template knobs (e.g. enable_thinking)
            # when tokenizing, otherwise `prompt_token_ids` (and therefore logged
            # `prompt_str`) can be built with different chat template settings than
            # the actual generation request.
            tokenize_body_dict = dict()
            for key in ("model", "messages", "tools", "chat_template_kwargs"):
                if key in body_dict:
                    tokenize_body_dict[key] = body_dict[key]

            # The base url has /v1 at the end but vLLM's tokenize endpoint does not have v1, hence the ..
            tokenize_response = await client.create_tokenize(**tokenize_body_dict)
            """
            END
            """

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                    # prompt_token_ids=chat_completion_dict["prompt_token_ids"],
                    prompt_token_ids=tokenize_response["tokens"],
                    # generation_token_ids=choice_dict["token_ids"],
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )
            _logprob_parity_dump("vllm", tokenize_response["tokens"], generation_token_ids, generation_log_probs)

            # Clean the duplicated information
            choice_dict.pop("logprobs")
            # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            # chat_completion_dict.pop("prompt_token_ids")
            # choice_dict.pop("token_ids")

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)

    def _resolve_client(self, request: Request) -> NeMoGymAsyncOpenAI:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        return client

    # =======================================================
    # SGLang engine path (see VLLMModelConfig.engine == "sglang")
    # =======================================================

    # Hermes tool-call format emitted by SGLang's --tool-call-parser hermes and the SWE
    # chat template: one or more <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
    # blocks. The capture is the JSON object, anchored by the closing tag (so nested braces
    # in arguments are handled without brace-balancing).
    _SGLANG_TOOL_CALL_PATTERN: ClassVar = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _SGLANG_ARGS_PATTERN: ClassVar = re.compile(r"\"arguments\"\s*:\s*(.*)\}\s*$", re.DOTALL)
    # Turn-terminating special tokens the chat template re-emits after an assistant message.
    # We strip them from the decoded generation so they are not doubled on history re-render.
    _SGLANG_EOS_MARKERS: ClassVar = ("<|im_end|>", "<|endoftext|>")

    def _get_sglang_tokenizer(self) -> Any:
        if self._sglang_tokenizer is None:
            from transformers import AutoTokenizer

            self._sglang_tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        return self._sglang_tokenizer

    def _get_sglang_chat_template(self) -> Optional[str]:
        if self._sglang_chat_template is None and self.config.sglang_chat_template_path:
            with open(self.config.sglang_chat_template_path) as f:
                self._sglang_chat_template = f.read()
        return self._sglang_chat_template

    def _full_sglang_tokenize(
        self, messages: List[Any], tools: Any, chat_template_kwargs: Dict[str, Any]
    ) -> List[int]:
        """Tokenize the full chat prompt via the chat template (the original, non-spliced path)."""
        tokenizer = self._get_sglang_tokenizer()
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=self._get_sglang_chat_template(),
            add_generation_prompt=True,
            tokenize=True,
            **chat_template_kwargs,
        )
        # transformers v5's apply_chat_template(tokenize=True) returns a BatchEncoding
        # (dict-like), not a flat list; normalize to a JSON-serializable List[int].
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(t) for t in encoded]

    def _sglang_eos_nl(self) -> List[int]:
        if self._sglang_eos_nl_ids is None:
            enc = self._get_sglang_tokenizer()("<|im_end|>\n", add_special_tokens=False)
            self._sglang_eos_nl_ids = [int(t) for t in enc["input_ids"]]
        return self._sglang_eos_nl_ids

    def _sglang_followup_fragment_ids(
        self, new_msgs: List[Any], chat_template_kwargs: Dict[str, Any]
    ) -> Optional[List[int]]:
        """Token ids for the new (non-assistant) messages + the next generation-prompt header,
        rendered as a standalone fragment that follows a prior assistant turn. Derived by
        differencing two template renders against an anchor assistant turn, then tokenizing the
        suffix. Safe to splice onto the running sequence because the splice boundary is the
        assistant turn's ``<|im_end|>\\n`` (a special token), across which byte-level BPE does
        not merge (validated). Returns None if the template is not splice-friendly -> caller
        falls back to a full re-tokenize."""
        tokenizer = self._get_sglang_tokenizer()
        ct = self._get_sglang_chat_template()
        anchor = [{"role": "assistant", "content": "X"}]
        try:
            full = tokenizer.apply_chat_template(
                anchor + list(new_msgs), tools=None, chat_template=ct,
                add_generation_prompt=True, tokenize=False, **chat_template_kwargs,
            )
            base = tokenizer.apply_chat_template(
                anchor, tools=None, chat_template=ct,
                add_generation_prompt=False, tokenize=False, **chat_template_kwargs,
            )
        except Exception:
            return None
        if not isinstance(full, str) or not isinstance(base, str) or not full.startswith(base):
            return None
        enc = tokenizer(full[len(base):], add_special_tokens=False)
        return [int(t) for t in enc["input_ids"]]

    @staticmethod
    def _sglang_msg_sig(m: Dict[str, Any]) -> Tuple[Any, str, str]:
        return (
            m.get("role"),
            json.dumps(m.get("content"), sort_keys=True, default=str),
            json.dumps(m.get("tool_calls"), sort_keys=True, default=str),
        )

    @classmethod
    def _sglang_messages_match(cls, a: List[Any], b: List[Any]) -> bool:
        return len(a) == len(b) and all(
            cls._sglang_msg_sig(x) == cls._sglang_msg_sig(y) for x, y in zip(a, b)
        )

    def _build_sglang_prompt_ids(
        self, request: Request, messages: List[Any], tools: Any,
        chat_template_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], Optional[str]]:
        """Return (prompt_token_ids, session_id). Splices the prior assistant turn's exact
        generation tokens when this is a continuation of a cached session; else full tokenize."""
        try:
            sid = request.session.get(SESSION_ID_KEY)
        except Exception:
            sid = None
        if sid is not None:
            state = self._sglang_session_seq.get(sid)
            if state is not None:
                prev = state["messages"]
                n = len(prev)
                if (
                    len(messages) > n
                    and messages[n].get("role") == "assistant"
                    and all(m.get("role") != "assistant" for m in messages[n + 1:])
                    and self._sglang_messages_match(messages[:n], prev)
                ):
                    frag = self._sglang_followup_fragment_ids(messages[n + 1:], chat_template_kwargs)
                    if frag is not None:
                        return state["seq"] + frag, sid
        return self._full_sglang_tokenize(messages, tools, chat_template_kwargs), sid

    def _update_sglang_session_seq(
        self, sid: Optional[str], messages: List[Any],
        prompt_token_ids: List[int], generation_token_ids: List[int],
    ) -> None:
        """Cache the running sequence through this assistant turn (prompt + gen + ``<|im_end|>\\n``)
        for the next turn's splice."""
        if sid is None:
            return
        eos_nl = self._sglang_eos_nl()  # e.g. [151645, 198]
        seq = list(prompt_token_ids) + list(generation_token_ids)
        if not seq or seq[-1] != eos_nl[0]:
            seq = seq + eos_nl
        else:
            seq = seq + eos_nl[1:]  # gen already ended with <|im_end|>; just add the trailing \n
        # Bound memory: refresh this sid's insertion order, evict oldest beyond the cap.
        # Evicted sessions simply fall back to a full tokenize on their next turn (safe).
        self._sglang_session_seq.pop(sid, None)
        while len(self._sglang_session_seq) >= 8192:
            self._sglang_session_seq.pop(next(iter(self._sglang_session_seq)), None)
        self._sglang_session_seq[sid] = {"messages": list(messages), "seq": seq}

    def _parse_sglang_generation(self, text: str) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        """Reconstruct (reasoning_content, content, tool_calls) from SGLang /generate raw text.

        The qwen3-thinking generation prompt ends with ``<think>\\n``, so the generated text
        begins INSIDE the reasoning block (no opening ``<think>``). Everything up to the first
        ``</think>`` is therefore reasoning; hermes tool calls are parsed out of the remainder.
        This mirrors what the SGLang server's reasoning_parser=qwen3-thinking +
        tool_call_parser=hermes would have produced on /v1/chat/completions, so downstream
        Responses marshaling is identical to the vLLM path.
        """
        reasoning_content: Optional[str] = None
        if self.config.uses_reasoning_parser and "</think>" in text:
            reasoning_content, _, remainder = text.partition("</think>")
        else:
            remainder = text

        tool_calls: List[Dict[str, Any]] = []
        for match in self._SGLANG_TOOL_CALL_PATTERN.finditer(remainder):
            block = match.group(1)
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            # Preserve the model's EXACT arguments serialization (function.arguments is a JSON
            # string in the OpenAI schema). Keeping the raw substring -- rather than
            # re-serializing the parsed dict -- means the chat template re-renders the assistant
            # turn byte-identically, which the nemo_gym.py contiguity assert depends on.
            args_match = self._SGLANG_ARGS_PATTERN.search(block)
            if args_match is not None:
                arguments = args_match.group(1).strip()
            else:
                arguments = json.dumps(parsed.get("arguments", {}))
            tool_calls.append(
                dict(
                    id=f"call_{uuid4().hex}",
                    type="function",
                    function=dict(name=parsed.get("name"), arguments=arguments),
                )
            )

        content = self._SGLANG_TOOL_CALL_PATTERN.sub("", remainder).strip()
        return reasoning_content, content, tool_calls

    async def _sglang_chat_completion(
        self, request: Request, body_dict: Dict[str, Any]
    ) -> NeMoGymChatCompletion:
        """SGLang v0.5.10 generation path (see VLLMModelConfig.engine).

        Tokenizes the chat-templated prompt locally and generates via SGLang's native
        /generate (return_logprob=True) -- the only v0.5.10 source of the exact sampled
        integer token ids AND their logprobs (needed for token-level RL + logprob parity).
        The decoded text is re-parsed into reasoning + hermes tool_calls so the returned
        object is shaped exactly like the vLLM /v1/chat/completions response, keeping every
        downstream Responses-API conversion identical.
        """
        client = self._resolve_client(request)
        _maybe_launch_vllm_parity_tf(self.config)

        messages = body_dict["messages"]
        if self.config.replace_developer_role_with_system:
            for message_dict in messages:
                if message_dict.get("role") == "developer":
                    message_dict["role"] = "system"
        tools = body_dict.get("tools")

        # Merge config chat_template_kwargs with per-request metadata overrides (mirrors
        # _preprocess_chat_completion_create_params so reasoning toggles behave identically).
        chat_template_kwargs: Dict[str, Any] = {}
        if self.config.chat_template_kwargs:
            chat_template_kwargs = deepcopy(self.config.chat_template_kwargs)
        metadata = body_dict.get("metadata", dict())
        chat_template_kwargs.update(json.loads(metadata.get("chat_template_kwargs", "{}")))

        tokenizer = self._get_sglang_tokenizer()  # used below to decode generation_token_ids
        # Build prompt token ids with contiguity-preserving splicing across turns (falls back
        # to a full chat-template tokenize on the first turn / cache miss / history condensation).
        prompt_token_ids, _splice_sid = self._build_sglang_prompt_ids(
            request, messages, tools, chat_template_kwargs
        )

        # Map the OpenAI sampling knobs onto SGLang /generate sampling_params.
        # spaces_between_special_tokens=False mirrors the NeMo-RL SGLang backend and keeps
        # special tokens (</think>, <tool_call>) tight in the decoded text we parse below.
        sampling_params: Dict[str, Any] = {"spaces_between_special_tokens": False}
        # max_tokens of None OR 0 means "unlimited" (the SWE agent sends max_output_tokens=0).
        # SGLang /generate would otherwise fall back to max_new_tokens=128 and truncate reasoning
        # before </think>; fill the remaining context instead (matches vLLM + the recipe).
        max_new_tokens = body_dict.get("max_tokens") or None
        if max_new_tokens is None and self.config.sglang_max_total_sequence_length:
            # Fill the remaining context, leaving a small margin: SGLang /generate rejects a
            # request whose input + max_new_tokens >= context_length (it requires strictly less,
            # unlike vLLM which allows ==). Reserve a few tokens to stay safely under the limit.
            max_new_tokens = self.config.sglang_max_total_sequence_length - len(prompt_token_ids) - 8
            max_new_tokens = max(1, max_new_tokens)
        if max_new_tokens:
            sampling_params["max_new_tokens"] = max_new_tokens
        for key in ("temperature", "top_p", "top_k", "stop"):
            if body_dict.get(key) is not None:
                sampling_params[key] = body_dict[key]

        gen = await client.create_generate(
            input_ids=prompt_token_ids,
            sampling_params=sampling_params,
            return_logprob=True,
        )

        meta_info = gen.get("meta_info") or {}
        # Each tuple is (logprob, token_id, ...). Sourcing both ids and logprobs from the SAME
        # list guarantees they are 1:1 aligned in count and order -- mirrors
        # nemo_rl/models/generation/sglang/sglang_generation.py:generate_one_sample.
        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        generation_token_ids = [item[1] for item in output_token_logprobs]
        generation_log_probs = [item[0] for item in output_token_logprobs]

        # Contiguity fix: cache the running token sequence through this assistant turn so the
        # next turn splices these EXACT generation_token_ids instead of re-tokenizing them.
        self._update_sglang_session_seq(
            _splice_sid, messages, prompt_token_ids, generation_token_ids
        )
        _logprob_parity_dump(
            "sglang", prompt_token_ids, generation_token_ids, generation_log_probs
        )

        # Decode the EXACT sampled ids with skip_special_tokens=False so reasoning markers
        # (</think> is a SPECIAL token, id 151668) survive. SGLang's gen["text"] decodes with
        # skip_special_tokens=True by default, which STRIPS </think> -> the reasoning never gets
        # wrapped -> the <think> wrapper is dropped on history re-render -> the nemo_gym.py:199
        # contiguity assert fires on every multi-turn rollout. spaces_between_special_tokens=False
        # keeps the markers tight. Then strip the trailing EOS the chat template re-adds.
        generated_text = tokenizer.decode(
            generation_token_ids, skip_special_tokens=False, spaces_between_special_tokens=False
        )
        _stripped = True
        while _stripped:
            _stripped = False
            generated_text = generated_text.rstrip("\n")
            for _eos in self._SGLANG_EOS_MARKERS:
                if generated_text.endswith(_eos):
                    generated_text = generated_text[: -len(_eos)]
                    _stripped = True
        reasoning_content, content, tool_calls = self._parse_sglang_generation(generated_text)

        if (meta_info.get("finish_reason") or {}).get("type") == "length":
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        # Re-embed reasoning into <think> tags and prepend to content, identical to the vLLM
        # reasoning-parser branch, so postprocess_assistant_message_dict re-extracts it.
        if self.config.uses_reasoning_parser and reasoning_content:
            content = self._converter._wrap_reasoning_in_think_tags([reasoning_content]) + (content or "")

        message_dict: Dict[str, Any] = dict(
            role="assistant",
            content=content or None,
            tool_calls=tool_calls or None,
        )
        if self.config.return_token_id_information:
            message_dict.update(
                dict(
                    prompt_token_ids=prompt_token_ids,
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

        chat_completion_dict = dict(
            id=f"chtcmpl-{uuid4().hex}",
            object="chat.completion",
            created=int(time()),
            model=self.config.model,
            choices=[
                dict(index=0, finish_reason=finish_reason, message=message_dict, logprobs=None)
            ],
            usage=dict(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(generation_token_ids),
                total_tokens=len(prompt_token_ids) + len(generation_token_ids),
            ),
        )
        return NeMoGymChatCompletion.model_validate(chat_completion_dict)


class VLLMConverterResponsesToChatCompletionsState(BaseModel):
    return_token_id_information: bool

    messages: List[NeMoGymChatCompletionMessageParam] = Field(default_factory=list)

    # We are mapping from Response input items to chat completions messages, which is many to one.
    # Our state will accumulate the reasoning, chat, and tool calls for assistant messages.
    content_buffer: str = ""  # Buffer for reasoning and chat
    tool_calls_buffer: List[NeMoGymChatCompletionMessageToolCallParam] = Field(default_factory=list)

    # Will only be populated if return_token_id_information is True.
    token_information: Optional[TokenIDLogProbMixin] = None

    def flush_assistant(self) -> None:
        if not (self.content_buffer or self.tool_calls_buffer):
            return

        shared_params = dict(
            content=self.content_buffer or None,
            role="assistant",
            tool_calls=self.tool_calls_buffer,
        )

        # We check here that self.token_information is non-empty since it's possible that some assistant messages are entirely inputs and are not generated by the model in this trajectory.
        if self.return_token_id_information and self.token_information:
            message = NeMoGymChatCompletionAssistantMessageForTrainingParam(
                **shared_params,
                **self.token_information.model_dump(),
            )
        else:
            message = NeMoGymChatCompletionAssistantMessageParam(**shared_params)

        self.messages.append(message)

        self.content_buffer = ""
        self.tool_calls_buffer = []


class VLLMConverter(BaseModel):
    return_token_id_information: bool

    # =======================================================
    # Reasoning handling. This may change across models and model families
    # =======================================================

    THINK_TAG_PATTERN: ClassVar = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{t}</think>" for t in texts if t)

    @classmethod
    def _parse_think_tags(cls, content: str) -> Tuple[List[str], str]:
        # Extract reasoning content from between <think></think> tags.
        matches = cls.THINK_TAG_PATTERN.findall(content)
        # Remove reasoning from main content
        cleaned = cls.THINK_TAG_PATTERN.sub("", content)
        return matches, cleaned

    # =======================================================
    # Response create params to Chat Completion create params
    # =======================================================

    def responses_to_chat_completion_create_params(
        self,
        responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        responses_create_params = responses_create_params.model_dump(exclude_unset=True)

        # Tracks messages including reasoning for each respective message type helper function
        state = VLLMConverterResponsesToChatCompletionsState(
            return_token_id_information=self.return_token_id_information
        )

        # Input can be a string. Wrap in a ResponseInput-like
        response_input = responses_create_params["input"]
        if isinstance(response_input, str):
            wrapped_input = {
                "content": [
                    {
                        "text": response_input,
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
            input_messages = [wrapped_input]
        else:
            input_messages = responses_create_params.pop("input", [])

        for m in input_messages:
            if not m.get("type") and m.get("role"):
                m["type"] = "message"

            match m["type"]:
                case "message":
                    self._format_message(m, state)
                case "reasoning":
                    self._format_reasoning(m, state)
                case "function_call":
                    self._format_function_call(m, state)
                case "function_call_output":
                    self._format_function_call_output(m, state)
                case _:  # pragma: no cover
                    raise NotImplementedError(f"Unsupported message type: {m}")

            if self.return_token_id_information and m.get("prompt_token_ids"):
                state.token_information = TokenIDLogProbMixin(
                    prompt_token_ids=m["prompt_token_ids"],
                    generation_token_ids=m["generation_token_ids"],
                    generation_log_probs=m["generation_log_probs"],
                )

        state.flush_assistant()

        model = responses_create_params.pop("model", None)
        if model is not None:
            responses_create_params["model"] = model

        # The corresponding parameter to `max_output_tokens`` is `max_tokens`
        max_output_tokens = responses_create_params.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            responses_create_params["max_tokens"] = max_output_tokens

        tools = responses_create_params.pop("tools", None)
        if tools is not None:
            responses_create_params["tools"] = []
            for tool_dict in tools:
                tool_dict = tool_dict.copy()
                tool_dict.pop("type", None)

                # As of vLLM 0.17.1, vLLM Chat Completions does not accept this `strict` parameter on tool definitions that OpenAI accepts.
                tool_dict.pop("strict", None)
                responses_create_params["tools"].append(
                    NeMoGymChatCompletionToolParam(type="function", function=NeMoGymFunctionDefinition(**tool_dict))
                )

        chat_completion_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=state.messages,
            **responses_create_params,
        )

        return chat_completion_create_params

    def _format_function_call_output(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        state.flush_assistant()

        assert "call_id" in m
        converted = NeMoGymChatCompletionToolMessageParam(
            content=m["output"],
            role="tool",
            tool_call_id=m["call_id"],
        )
        state.messages.append(converted)

    def _format_message(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        content = m["content"]

        if isinstance(content, list) and m["role"] != "assistant":
            converted_parts = []
            for part_param in content:
                match part_param["type"]:
                    case "input_text":
                        converted_parts.append({"type": "text", "text": part_param["text"]})
                    case "input_image":
                        image_url = part_param.get("image_url", "")
                        detail = part_param.get("detail", "auto")
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                        )
                    case _:
                        raise NotImplementedError(f"Unsupported part param type: {part_param['type']}")
            content = converted_parts
            m["content"] = content

        match m["role"]:
            case "assistant":
                # Handle reasoning
                final_content = ""
                if isinstance(m["content"], list):
                    content_str = "".join([part.get("text", "") for part in m["content"]])
                    final_content += content_str
                elif isinstance(m["content"], str):
                    final_content += m["content"]
                else:
                    raise NotImplementedError(
                        f"Expected m['content'] to be str or list[dict], but got {type(m['content']).__name__!r}: {m['content']!r}"
                    )

                converted = []
                state.content_buffer += final_content
            case "user":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionUserMessageParam(
                        content=content,
                        role="user",
                    )
                ]
            # TODO: Revisit this in case we need separate handling. Not all chat templates may support the 'developer' role.
            case "system":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionSystemMessageParam(
                        content=content,
                        role="system",
                    )
                ]
            case "developer":
                state.flush_assistant()
                converted = [
                    NeMoGymChatCompletionDeveloperMessageParam(
                        content=content,
                        role="developer",
                    )
                ]
            case _:  # pragma: no cover
                raise NotImplementedError(f"Unrecognized role for message: `{m['role']}`")

        state.messages.extend(converted)

    def _format_reasoning(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        """
        Collects text from 'reasoning' messages in responses api and appends it to a buffer.

        This is done to group together one (or multiple) reasoning message(s) into a single,
        cohesive block, later prepending it to a subsequent assistant message.
        See: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/how-to-faq.md#faq-openai-responses-vs-chat-completions-api for an example of reasoning in responses api.
        """
        if "summary" in m and m["summary"]:
            texts = [s["text"] for s in m["summary"]]
            state.content_buffer += self._wrap_reasoning_in_think_tags(texts)

    def _format_function_call(
        self,
        m: dict,
        state: VLLMConverterResponsesToChatCompletionsState,
    ) -> None:
        assert "call_id" in m
        tool_call = NeMoGymChatCompletionMessageToolCallParam(
            id=m["call_id"],
            function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                arguments=m["arguments"],
                name=m["name"],
            ),
            type="function",
        )
        state.tool_calls_buffer.append(tool_call)

    # =======================================================
    # Chat Completion to Response
    # =======================================================

    def postprocess_chat_response(self, choice: NeMoGymChoice) -> List[NeMoGymResponseOutputItem]:
        return self.postprocess_assistant_message_dict(choice.message.model_dump())

    def postprocess_assistant_message_dict(self, message_dict: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        response_output = []

        content = message_dict.get("content") or ""
        reasoning_matches, content = self._extract_reasoning_from_content(content)
        if reasoning_matches:
            reasoning_item = NeMoGymResponseReasoningItem(
                id=f"rs_{uuid4().hex}",
                type="reasoning",
                summary=[
                    NeMoGymSummary(text=reasoning_text, type="summary_text") for reasoning_text in reasoning_matches
                ],
                status="completed",
            )
            response_output.append(reasoning_item)

        tool_calls_raw = message_dict.get("tool_calls", []) or []
        # We need to return at least one output item. When the model decides to just stop with no chat or tool calls
        # We just add an output item with empty or null content here. This is prevalent e.g. in the case of base models that may not be the most reliable since they have not been instruction tuned.
        has_empty_output = not (response_output or tool_calls_raw)

        if content or has_empty_output:
            response_output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role=message_dict.get("role"),
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=content,
                            annotations=[],
                        )
                    ],
                    status="completed",
                    type="message",
                )
            )

        for tc in tool_calls_raw:
            assert "id" in tc
            response_output.append(
                NeMoGymResponseFunctionToolCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    call_id=tc["id"],
                    type="function_call",
                    status="completed",
                    id=tc["id"],
                )
            )

        # `"prompt_token_ids" in raw_message`: sometimes the model endpoint may go out of context length, in which case we return an empty response
        # In these cases, there are no token id information provided.
        if self.return_token_id_information and "prompt_token_ids" in message_dict:
            last_response_output_item = response_output[-1]
            train_cls = RESPONSES_TO_TRAIN[last_response_output_item.__class__]
            response_output[-1] = train_cls(
                **last_response_output_item.model_dump(),
                prompt_token_ids=message_dict["prompt_token_ids"],
                generation_token_ids=message_dict["generation_token_ids"],
                generation_log_probs=message_dict["generation_log_probs"],
            )

        return response_output

    def _extract_reasoning_from_content(self, content: str) -> Tuple[List[str], str]:
        # TODO: Currently only parses reasoning wrapped in <think>...</think> tags.
        # Maybe parameterize to support other model formats in the future.
        return self._parse_think_tags(content)

    def chat_completions_messages_to_responses_items(
        self, messages: List[Dict[str, Any]]
    ) -> List[NeMoGymResponseOutputItem]:
        output_items = []

        for message in messages:
            role = message["role"]
            if role in ("user", "system", "developer"):
                output_items.append(NeMoGymEasyInputMessage.model_validate(message))
            elif role == "assistant":
                output_items.extend(self.postprocess_assistant_message_dict(message))
            elif role == "tool":
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                        status="completed",
                    )
                )
            else:
                raise NotImplementedError(f"Unrecognized role: {role}!")

        return output_items


def split_responses_input_output_items(
    items: List[NeMoGymResponseOutputItem],
) -> Tuple[List[NeMoGymResponseOutputItem], List[NeMoGymResponseOutputItem]]:
    if not items:
        return [], []

    for i, item in enumerate(items):
        if getattr(item, "role", None) == "assistant" or getattr(item, "type", None) in {
            "reasoning",
            "reasoning_item",
        }:
            break

    return items[:i], items[i:]


if __name__ == "__main__":
    VLLMModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = VLLMModel.run_webserver()  # noqa: F401
