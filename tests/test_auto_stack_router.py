from unittest.mock import MagicMock, patch

import pytest

from src.auto_stack_router import (
    AutoStackNotReady,
    build_model_pool,
    check_auto_stack_ready,
    hint_for_agent_round,
    hint_for_chat,
    is_auto_stack_model,
    resolve_model_on_endpoint,
)
from src.constants import AUTO_STACK_MODEL_ID as CONST_ID


def test_is_auto_stack_model():
    assert is_auto_stack_model(CONST_ID)
    assert not is_auto_stack_model("qwen3:8b")


def test_hint_for_chat_lookup():
    assert hint_for_chat("what is JWT?") == "lookup"


def test_hint_for_agent_code_tools():
    assert hint_for_agent_round(prompt="fix this", relevant_tools={"shell"}) == "code"


@patch("src.auto_stack_router._load_endpoint")
def test_resolve_model_on_endpoint_scoped(mock_load):
    ep = MagicMock()
    ep.base_url = "http://127.0.0.1:11434"
    ep.name = "Ollama"
    ep.api_key = ""
    mock_load.return_value = ep
    with patch("src.auto_stack_router._endpoint_enabled_models", return_value=["gemma4:e4b", "qwen3:8b"]):
        with patch("src.auto_stack_router.build_chat_url", return_value="http://127.0.0.1:11434/api/chat"):
            with patch("src.auto_stack_router.build_headers", return_value={"Authorization": "Bearer x"}):
                url, model, headers = resolve_model_on_endpoint(
                    "qwen3:8b",
                    endpoint_url="http://127.0.0.1:11434",
                    headers={"X-Test": "1"},
                    owner=None,
                )
    assert model == "qwen3:8b"
    assert "11434" in url
    assert headers.get("X-Test") == "1"


@patch("src.auto_stack_router.get_setting")
@patch("src.auto_stack_router.load_local_llm_router")
@patch("src.auto_stack_router.installed_tags_for_endpoint")
def test_build_model_pool_intersection(mock_installed, mock_load_ss, mock_setting):
    mock_installed.return_value = ["gemma4:e4b", "qwen3:8b", "qwen3:14b"]
    mock_setting.side_effect = lambda key, default=None: {
        "auto_stack_vram_gb": 16,
        "auto_stack_quant": "qat",
        "auto_stack_models": [],
    }.get(key, default)
    ss = MagicMock()
    ss.recommended_models.return_value = ["gemma4:e4b", "qwen3:8b", "qwen3:14b", "deepseek-r1:8b"]
    mock_load_ss.return_value = ss
    pool = build_model_pool("http://127.0.0.1:11434")
    assert "gemma4:e4b" in pool
    assert len(pool) >= 2


@patch("src.auto_stack_router.get_setting")
@patch("src.auto_stack_router.load_local_llm_router")
@patch("src.auto_stack_router.installed_tags_for_endpoint")
def test_build_model_pool_falls_back_to_installed_only(mock_installed, mock_load_ss, mock_setting):
    mock_installed.return_value = ["qwen3:8b", "qwen3:14b", "llama3.2:3b"]
    mock_setting.side_effect = lambda key, default=None: {
        "auto_stack_vram_gb": 16,
        "auto_stack_quant": "qat",
        "auto_stack_models": [],
    }.get(key, default)
    ss = MagicMock()
    ss.recommended_models.return_value = ["gemma4:e4b", "qwen3:8b"]
    mock_load_ss.return_value = ss
    pool = build_model_pool("http://127.0.0.1:11434")
    assert pool == ["qwen3:8b", "qwen3:14b", "llama3.2:3b"]


@patch("src.auto_stack_router.installed_tags_for_endpoint")
def test_check_auto_stack_ready_no_models(mock_installed):
    mock_installed.return_value = []
    with pytest.raises(AutoStackNotReady) as exc:
        check_auto_stack_ready("http://127.0.0.1:11434/v1")
    assert exc.value.code == "no_models"
    assert "Cookbook" in str(exc.value)


@patch("src.auto_stack_router.installed_tags_for_endpoint")
def test_check_auto_stack_ready_one_model(mock_installed):
    mock_installed.return_value = ["qwen3:8b"]
    with pytest.raises(AutoStackNotReady) as exc:
        check_auto_stack_ready("http://127.0.0.1:11434/v1")
    assert exc.value.code == "insufficient_models"
    assert "qwen3:8b" in str(exc.value)


def test_check_auto_stack_ready_no_endpoint():
    with pytest.raises(AutoStackNotReady) as exc:
        check_auto_stack_ready("")
    assert exc.value.code == "no_endpoint"


def test_match_tag_no_fuzzy_cross_model():
    from src.auto_stack_router import _match_tag

    assert _match_tag("gemma4:e4b", ["qwen3:8b", "qwen3:14b"]) is None
    assert _match_tag("qwen3:8b", ["qwen3:8b", "qwen3:14b"]) == "qwen3:8b"
