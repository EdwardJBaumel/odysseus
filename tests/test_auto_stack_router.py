from unittest.mock import MagicMock, patch

import pytest

from src.auto_stack_router import (
    AutoStackNotReady,
    AutoStackResolution,
    build_model_pool,
    check_auto_stack_ready,
    is_auto_stack_model,
    resolve_auto_stack,
    resolve_model_on_endpoint,
)
from src.constants import AUTO_STACK_MODEL_ID as CONST_ID


def test_is_auto_stack_model():
    assert is_auto_stack_model(CONST_ID)
    assert not is_auto_stack_model("qwen3:8b")


@patch("src.auto_stack_router.resolve_model_on_endpoint")
@patch("src.auto_stack_router.build_model_pool")
@patch("src.auto_stack_router.load_local_llm_router")
def test_resolve_auto_stack_passes_mode_and_reasons(mock_load_ss, mock_pool, mock_resolve):
    mock_pool.return_value = ["gemma4:e4b", "qwen3:8b"]
    decision = MagicMock()
    decision.tier = MagicMock(value="simple")
    decision.model = "gemma4:e4b"
    decision.reasons = ("mode=agent", "keyword/heuristic scoring → tier simple")
    router = MagicMock()
    router.explain.return_value = decision
    mock_load_ss.return_value = router
    mock_resolve.return_value = ("http://127.0.0.1:11434/api/chat", "gemma4:e4b", {})

    res = resolve_auto_stack(
        prompt="Reply with exactly: pong",
        endpoint_url="http://127.0.0.1:11434",
        headers={},
        mode="agent",
    )

    router.configure.assert_called_once()
    router.explain.assert_called_once_with("Reply with exactly: pong", mode="agent")
    assert isinstance(res, AutoStackResolution)
    assert res.model == "gemma4:e4b"
    assert res.route_reasons == ("mode=agent", "keyword/heuristic scoring → tier simple")


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


@patch("src.auto_stack_router.resolve_auto_stack")
def test_resolve_task_endpoint_concrete_auto_stack(mock_resolve_stack):
    from src.auto_stack_router import AutoStackResolution
    from src.constants import AUTO_STACK_MODEL_ID
    from src.task_endpoint import _resolve_auto_stack_fallback

    mock_resolve_stack.return_value = AutoStackResolution(
        endpoint_url="http://127.0.0.1:11434/api/chat",
        model="gemma4:e4b",
        headers={},
        tier="simple",
        route_reasons=("mode=chat",),
        pool=("gemma4:e4b", "qwen3:8b"),
    )
    url, model, headers = _resolve_auto_stack_fallback(
        "http://127.0.0.1:11434/v1",
        AUTO_STACK_MODEL_ID,
        {},
    )
    assert model == "gemma4:e4b"
    assert "11434" in url


def test_resolve_task_endpoint_passthrough_non_auto():
    from src.task_endpoint import _resolve_auto_stack_fallback

    url, model, headers = _resolve_auto_stack_fallback(
        "http://127.0.0.1:11434/v1",
        "qwen3:8b",
        {"X": "1"},
    )
    assert model == "qwen3:8b"
    assert headers == {"X": "1"}
