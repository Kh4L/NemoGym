# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


@pytest.mark.parametrize(
    "base_url",
    ["http://router:8000", "http://router:8000/", "http://router:8000/v1", "http://router:8000/v1/"],
)
async def test_generate_normalizes_router_url_and_uses_shared_auth(monkeypatch, base_url: str) -> None:
    result = {"meta_info": {"output_token_logprobs": [[-0.5, 12, "a"]]}}
    response = SimpleNamespace(status=200, ok=True, read=AsyncMock(return_value=json.dumps(result).encode()))
    request = AsyncMock(return_value=response)
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    client = NeMoGymAsyncOpenAI(
        base_url=base_url,
        api_key="test-only",
        default_headers={"X-Route": "policy"},
        max_connection_retries=2,
    )
    assert await client.create_generate(input_ids=[1, 2], return_logprob=True) == result
    request.assert_awaited_once_with(
        method="POST",
        url="http://router:8000/generate",
        json={"input_ids": [1, 2], "return_logprob": True},
        headers={"X-Route": "policy", "Authorization": "Bearer test-only"},
        _internal=False,
        _max_connection_retries=2,
    )


async def test_generate_propagates_native_http_error(monkeypatch) -> None:
    response = SimpleNamespace(status=400, ok=False)
    request = AsyncMock(return_value=response)
    status_error = RuntimeError("native context-length error")
    check_status = AsyncMock(side_effect=status_error)
    monkeypatch.setattr("nemo_gym.openai_utils.request", request)
    monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", check_status)
    client = NeMoGymAsyncOpenAI(base_url="http://router:8000/v1", api_key="test-only")
    with pytest.raises(RuntimeError, match="native context-length error") as raised:
        await client.create_generate(input_ids=[1])
    assert raised.value is status_error
    check_status.assert_awaited_once_with(response)
