# SGLang model server

This Responses-API model server connects NeMo Gym to an SGLang server managed
outside Gym. It preserves the exact prompt IDs, sampled token IDs, and sampled
token logprobs required by token-level RL training.

## Transports

### `transport: chat` (default) — requires **sglang >= 0.5.13**

Drives SGLang's OpenAI-compatible `/v1/chat/completions`, requesting the training
metadata through SGLang's native `return_meta_info` / `return_prompt_token_ids`
extensions. Token IDs and logprobs come back on each choice as
`meta_info.output_token_logprobs` and `prompt_token_ids`.

This is the stable token-ID/logprob contract the `/generate` path was written to
wait for: it landed with the sglang-miles TITO sync series
([sgl-project/sglang#23751](https://github.com/sgl-project/sglang/pull/23751), merged
before the 0.5.13 cut) and is in the released tree, so **no patched build or fork is
required**.

Prefer it whenever the server supports it. Only token extraction is overridden, so
prompt templating, **tool-call parsing**, sampling parameters, auth, and
context-overflow handling are all done server-side — there is no local tokenizer that
can drift from the server's, and no client-side tool parser to keep in sync.

### `transport: generate` — for builds without chat-side TITO

Renders the chat template locally and calls SGLang's native `/generate` with
`return_logprob=true`. Required for forks that predate the TITO series (for example
those serving diffusion LLMs). This path carries the session-splice and
context-overflow rules below, and parses tool calls client-side because `/generate`
returns raw text.

In both transports, a generation SGLang reports as `finish_reason="abort"` raises
rather than being emitted, so a server-cancelled partial cannot enter a training
batch looking like a completed turn.

## Multi-turn session splicing (`generate` transport)

For a multi-turn session, the adapter caches the token sequence and splices
each prior assistant turn's exact sampled IDs into the next prompt. It never
re-tokenizes those sampled turns. Tools and chat-template kwargs must therefore
remain fixed for the life of a session; the adapter fails loudly if they
change. If a prompt already fills `context_length`, the adapter returns a
terminal response with `finish_reason="length"` instead of truncating the
prefix. Both behaviors preserve the trainer's prefix contiguity invariant.

The session cache is process-local. Run one Gym worker per model-server
instance, or provide sticky routing that keeps every turn of a session on the
same worker.

## Configuration

See `configs/sglang_model_for_training.yaml`.

See `configs/sglang_model_for_training.yaml` (chat) and
`configs/sglang_model_for_training_generate.yaml` (generate).

- `transport`: `chat` (default) or `generate`.
- `base_url`: for `chat` it must end in `/v1`, exactly like `vllm_model`. For
  `generate` either form works — `create_generate` strips a trailing `/v1` itself.
- `model`: a tokenizer/model identifier available to the Gym server.

Everything below applies to `transport: generate` only; on the chat path SGLang
owns these concerns.

- `context_length`: **required** for `generate` — the SGLang server's total context
  limit. A locally tokenized prompt is unbounded unless this server bounds it, so
  the example config leaves it mandatory (`???`) rather than defaulting, so a
  mismatched limit cannot be selected silently. Unused (and unset) for `chat`.
- `sglang_chat_template`: optional inline copy of the server/training template.
- `sglang_chat_template_path`: optional path to the exact server chat template.
- `sglang_tool_format`: `hermes` or `qwen3_coder`.
- `sglang_eos_markers` / `sglang_turn_suffix`: end-of-turn markers, defaulting to
  ChatML. A model whose template closes turns differently **must** override both, or
  the splice writes a malformed turn boundary into every follow-up prompt.
- `trust_remote_code`: forwarded to the local tokenizer loader; defaults to `false`.

The leaf package pins `transformers`, matching NeMo RL's SGLang worker environment,
because this path renders the template and tokenizes locally — a tokenizer or
template delta between adapter and server surfaces as a rollout contiguity failure
rather than an import error.

## CPU-only tests

From the Gym checkout:

```bash
uv run --extra dev pytest responses_api_models/sglang_model/tests
```

The direct tests mock the tokenizer and SGLang HTTP client; they do not load
weights or require a GPU.
