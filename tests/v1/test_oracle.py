# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def test_unsupported_configs():
    with pytest.raises(ValueError):
        AsyncEngineArgs(
            model=MODEL,
            speculative_config={
                "model": MODEL,
            },
        ).create_engine_config()


def test_v2_model_runner_requires_uva(monkeypatch: pytest.MonkeyPatch):
    """MRV2 falls back to the V1 model runner when UVA is unavailable
    (e.g. WSL2, where pinned memory is disabled by default)."""
    import vllm.config.vllm as vllm_config_module

    config = AsyncEngineArgs(model="Qwen/Qwen3-0.6B").create_engine_config()
    assert config.use_v2_model_runner

    monkeypatch.setattr(vllm_config_module, "is_uva_available", lambda: False)
    assert not config.use_v2_model_runner
