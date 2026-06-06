from types import SimpleNamespace

import pytest

from src.agent_model_mode import compute_is_api_model


def test_deepseek_r1_not_api_model():
    assert compute_is_api_model("http://127.0.0.1:8000/v1", "deepseek-r1:8b") is False


def test_openai_host_is_api_model():
    assert compute_is_api_model("https://api.openai.com/v1/chat/completions", "gpt-4o") is True
