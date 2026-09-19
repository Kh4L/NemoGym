# SGLang model server

This adapter uses SGLang's native `/generate` endpoint and returns exact sampled
token IDs and aligned log probabilities through Gym's normal Responses API.
It renders the prompt locally, preserves sampled IDs across session-bound tool
turns, and rejects malformed token metadata or unverifiable continuation splices.

Use `configs/sglang_model_for_training.yaml`. Set `context_length` to the policy
server's total context limit, and use the same tokenizer, chat template, and
revision as that server (`tokenizer` can specify a local tokenizer path).
`base_url` accepts either the server/router root or its `/v1` endpoint, with or
without a trailing slash. `sampling_overrides` wins over per-request sampling
controls, while the remaining context always caps generation length.

Tool formats are `hermes` (JSON `<tool_call>` blocks) and `qwen3_coder` (XML-like
function/parameter blocks). The default end-of-turn markers and suffix are
ChatML; other templates must set `sglang_eos_markers` and `sglang_turn_suffix`.
Thinking text is processed through Gym's existing Responses converter.

Token echo and Gym's normal capture middleware are retained. Worker-owned
`token_id_capture.external_staging` is not supported by this native endpoint and
fails at startup; enabled capture requires `return_token_id_information: true`.
The vLLM adapter and its capture protocol are unchanged. Native Responses,
completions transport, assistant prefills, and engine-specific prefix-token
extensions are rejected rather than silently approximated.

Unit tests use a fake tokenizer and mocked native responses:

```bash
pytest responses_api_models/sglang_model/tests tests/unit_tests/test_sglang_http_client.py
```

These unit tests do not substitute for a real policy-server rollout.
