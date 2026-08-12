# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-free helpers for the SGLang model server."""

import math
from numbers import Integral, Real
from typing import Any, Dict, List, Tuple


def _normalize_token_id(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(f"Malformed SGLang {field} token ID: expected an integer, got {value!r}")
    return int(value)


def _normalize_token_ids(values: Any, field: str) -> List[int]:
    if not isinstance(values, (list, tuple)):
        raise RuntimeError(f"Malformed SGLang {field} field: expected an array, got {type(values).__name__}")
    return [_normalize_token_id(value, field) for value in values]


def _normalize_logprob(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"Malformed SGLang generated-token logprob: expected a number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeError(f"Malformed SGLang generated-token logprob: expected a finite number, got {value!r}")
    return normalized


def _normalize_logprobs(values: Any, field: str) -> List[float]:
    if not isinstance(values, (list, tuple)):
        raise RuntimeError(f"Malformed SGLang {field} field: expected an array, got {type(values).__name__}")
    return [_normalize_logprob(value) for value in values]


def _has_visible_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    return True


def _validate_empty_generation(
    result: Dict[str, Any],
    meta: Dict[str, Any],
    output_ids: List[int] | None,
) -> None:
    """Require every server-visible output signal to agree on zero tokens."""
    visible = _has_visible_value(result.get("text")) or bool(output_ids)
    message = result.get("message")
    if isinstance(message, dict):
        visible = visible or any(
            _has_visible_value(message.get(key)) for key in ("content", "reasoning_content", "reasoning", "tool_calls")
        )

    completion_tokens = meta.get("completion_tokens")
    if completion_tokens is not None:
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, Integral):
            raise RuntimeError(
                f"Malformed SGLang completion_tokens field: expected an integer, got {completion_tokens!r}"
            )
        visible = visible or completion_tokens != 0

    if visible:
        raise RuntimeError(
            "SGLang returned visible output with empty generated-token metadata; "
            "refusing to emit an empty training loss mask"
        )


def _extract_output_ids(
    result: Dict[str, Any],
    meta: Dict[str, Any],
) -> List[int] | None:
    meta_output_ids = meta.get("output_ids")
    result_output_ids = result.get("output_ids")
    normalized_meta = (
        _normalize_token_ids(meta_output_ids, "meta_info.output_ids") if meta_output_ids is not None else None
    )
    normalized_result = (
        _normalize_token_ids(result_output_ids, "output_ids") if result_output_ids is not None else None
    )
    if normalized_meta is not None and normalized_result is not None and normalized_meta != normalized_result:
        raise RuntimeError(
            "SGLang returned conflicting output_ids fields: "
            f"meta_info.output_ids={normalized_meta!r}, output_ids={normalized_result!r}"
        )
    return normalized_meta if normalized_meta is not None else normalized_result


def _validate_selected_ids(
    selected_ids: List[int],
    output_ids: List[int] | None,
) -> None:
    if output_ids is not None and selected_ids != output_ids:
        raise RuntimeError(
            "SGLang returned generated-token IDs that do not match output_ids: "
            f"selected={selected_ids!r}, output_ids={output_ids!r}"
        )


def extract_generated_tokens_and_logprobs(
    result: Dict[str, Any],
) -> Tuple[List[int], List[float]]:
    """Return aligned generated token IDs and logprobs from ``/generate``.

    SGLang releases have emitted the selected-token data as dictionaries,
    tuples, or parallel value/index arrays. Missing or malformed data is a
    hard error because silently returning empty arrays would invalidate the
    training loss mask.
    """
    raw_meta = result.get("meta_info")
    if raw_meta is not None and not isinstance(raw_meta, dict):
        raise RuntimeError(f"Malformed SGLang meta_info field: expected an object, got {type(raw_meta).__name__}")
    meta = raw_meta or {}
    output_ids = _extract_output_ids(result, meta)
    structured_present = "output_token_logprobs" in meta
    split_present = any(
        key in source
        for source in (meta, result)
        for key in ("output_token_logprobs_val", "output_token_logprobs_idx")
    )
    if structured_present and split_present:
        raise RuntimeError(
            "SGLang returned conflicting generated-token logprob encodings: "
            "output_token_logprobs and split value/index arrays"
        )

    if structured_present:
        entries = meta["output_token_logprobs"]
        if not isinstance(entries, (list, tuple)):
            raise RuntimeError(
                f"Malformed SGLang output_token_logprobs field: expected an array, got {type(entries).__name__}"
            )
        if not entries:
            _validate_selected_ids([], output_ids)
            _validate_empty_generation(result, meta, output_ids)
            return [], []
        token_ids: List[int] = []
        logprobs: List[float] = []
        for entry in entries:
            if isinstance(entry, dict):
                token_id = entry.get("token_id", entry.get("id"))
                logprob = entry.get("logprob")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                logprob, token_id = entry[0], entry[1]
            else:
                raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {entry!r}")
            if token_id is None or logprob is None:
                raise RuntimeError(f"Malformed SGLang output_token_logprobs entry: {entry!r}")
            token_ids.append(_normalize_token_id(token_id, "output_token_logprobs"))
            logprobs.append(_normalize_logprob(logprob))
        _validate_selected_ids(token_ids, output_ids)
        return token_ids, logprobs

    meta_values_present = "output_token_logprobs_val" in meta
    result_values_present = "output_token_logprobs_val" in result
    values_present = meta_values_present or result_values_present
    meta_values = (
        _normalize_logprobs(meta["output_token_logprobs_val"], "meta_info.output_token_logprobs_val")
        if meta_values_present
        else None
    )
    result_values = (
        _normalize_logprobs(result["output_token_logprobs_val"], "output_token_logprobs_val")
        if result_values_present
        else None
    )
    if meta_values is not None and result_values is not None and meta_values != result_values:
        raise RuntimeError("SGLang returned conflicting output_token_logprobs_val fields")
    values = meta_values if meta_values is not None else result_values

    meta_indexes_present = "output_token_logprobs_idx" in meta
    result_indexes_present = "output_token_logprobs_idx" in result
    meta_indexes = (
        _normalize_token_ids(meta["output_token_logprobs_idx"], "meta_info.output_token_logprobs_idx")
        if meta_indexes_present
        else None
    )
    result_indexes = (
        _normalize_token_ids(result["output_token_logprobs_idx"], "output_token_logprobs_idx")
        if result_indexes_present
        else None
    )
    if meta_indexes is not None and result_indexes is not None and meta_indexes != result_indexes:
        raise RuntimeError("SGLang returned conflicting output_token_logprobs_idx fields")
    indexes = meta_indexes if meta_indexes is not None else result_indexes

    if values_present:
        selected_ids = indexes if indexes is not None else output_ids
        if not values:
            normalized_ids = (
                _normalize_token_ids(selected_ids, "output_token_logprobs_idx") if selected_ids is not None else []
            )
            if normalized_ids:
                raise RuntimeError(
                    f"SGLang returned mismatched generation fields: {len(normalized_ids)} ids for 0 logprobs"
                )
            _validate_selected_ids(normalized_ids, output_ids)
            _validate_empty_generation(result, meta, output_ids)
            return [], []
        if not selected_ids or len(selected_ids) != len(values):
            id_count = len(selected_ids) if selected_ids is not None else 0
            raise RuntimeError(
                f"SGLang returned mismatched generation fields: {id_count} ids for {len(values)} logprobs"
            )
        normalized_ids = list(selected_ids)
        _validate_selected_ids(normalized_ids, output_ids)
        return normalized_ids, list(values)

    result_keys = sorted(result)
    meta_keys = sorted(meta)
    raise RuntimeError(
        "SGLang /generate returned no generated-token logprobs "
        f"(result keys={result_keys}, meta_info keys={meta_keys}). "
        "Ensure return_logprob=true is supported and honored."
    )
