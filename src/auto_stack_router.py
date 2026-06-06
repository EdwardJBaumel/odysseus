"""Local-LLM-Router (split-stack) — Auto Select routing for local multi-model sessions."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from src.constants import AUTO_STACK_MODEL_ID
from src.endpoint_resolver import (
    _endpoint_enabled_models,
    build_chat_url,
    build_headers,
    normalize_base,
)
from src.settings import get_setting
from src.split_stack_runtime import load_local_llm_router
from src.teacher_escalation import is_self_hosted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoStackResolution:
    endpoint_url: str
    model: str
    headers: dict
    tier: str
    route_reasons: tuple[str, ...]
    pool: tuple[str, ...]


def is_auto_stack_model(model: str | None) -> bool:
    return (model or "").strip() == AUTO_STACK_MODEL_ID


def is_auto_stack_session(sess, *, require_enabled: bool = False) -> bool:
    """True when the session model is the Auto Select sentinel."""
    return is_auto_stack_model(getattr(sess, "model", None))


def is_auto_stack_active(sess) -> bool:
    """True when we should route via split-stack for this session."""
    if not is_auto_stack_session(sess):
        return False
    if not is_self_hosted(getattr(sess, "endpoint_url", "") or ""):
        return False
    return True


def _match_tag(requested: str, models: list[str]) -> str | None:
    if not requested or not models:
        return None
    if requested in models:
        return requested
    req_base = os.path.basename(requested.rstrip("/")).lower()
    for mid in models:
        if mid.lower() == requested.lower():
            return mid
        if os.path.basename(mid.rstrip("/")).lower() == req_base:
            return mid
    return None


def _load_endpoint(*, endpoint_url: str, owner: str | None):
    from core.database import ModelEndpoint, SessionLocal
    from src.auth_helpers import owner_filter

    session_base = normalize_base(endpoint_url or "")
    if not session_base:
        return None
    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            q = owner_filter(q, ModelEndpoint, owner)
        for ep in q.all():
            try:
                if normalize_base(ep.base_url or "") == session_base:
                    return ep
            except Exception:
                continue
        return None
    finally:
        db.close()


def installed_tags_for_endpoint(
    endpoint_url: str,
    *,
    owner: str | None = None,
) -> list[str]:
    ep = _load_endpoint(endpoint_url=endpoint_url, owner=owner)
    if ep is None:
        return []
    return list(_endpoint_enabled_models(ep))


class AutoStackNotReady(ValueError):
    """Auto (Local LLMs) cannot route — missing endpoint models or pool too small."""

    def __init__(self, message: str, *, code: str = "not_ready"):
        self.code = code
        super().__init__(message)


def check_auto_stack_ready(
    endpoint_url: str,
    *,
    owner: str | None = None,
) -> None:
    """Raise :class:`AutoStackNotReady` with an actionable message when routing cannot run."""
    if not (endpoint_url or "").strip():
        raise AutoStackNotReady(
            "Auto (Local LLMs) needs a local endpoint. "
            "Add Ollama (or another local server) in Settings or Cookbook, then try again.",
            code="no_endpoint",
        )
    installed = installed_tags_for_endpoint(endpoint_url, owner=owner)
    if not installed:
        raise AutoStackNotReady(
            "No models are installed on your local endpoint yet. "
            "Open Cookbook to pull at least 2 models, then refresh endpoints.",
            code="no_models",
        )
    if len(installed) < 2:
        only = installed[0]
        raise AutoStackNotReady(
            f"Auto (Local LLMs) needs at least 2 local models to route between; "
            f"you have {len(installed)} ({only}). Open Cookbook to add another model.",
            code="insufficient_models",
        )


def resolve_model_on_endpoint(
    model_tag: str,
    *,
    endpoint_url: str,
    headers: dict | None,
    owner: str | None = None,
) -> tuple[str, str, dict]:
    """Map a model tag to (chat_url, model_id, headers) on the session endpoint."""
    ep = _load_endpoint(endpoint_url=endpoint_url, owner=owner)
    if ep is None:
        raise ValueError(f"No enabled endpoint matches {endpoint_url!r}")
    enabled = _endpoint_enabled_models(ep)
    matched = _match_tag(model_tag, enabled)
    if not matched:
        raise ValueError(
            f"Model {model_tag!r} not found on endpoint {getattr(ep, 'name', endpoint_url)!r}. "
            f"Installed: {', '.join(enabled[:8])}{'...' if len(enabled) > 8 else ''}"
        )
    base = normalize_base(ep.base_url)
    chat_url = build_chat_url(base)
    hdrs = build_headers(ep.api_key, base)
    if headers:
        hdrs.update(headers)
    return chat_url, matched, hdrs


def _detect_vram_gb() -> int:
    manual = int(get_setting("auto_stack_vram_gb", 0) or 0)
    if manual > 0:
        return manual
    try:
        from services.hwfit.hardware import detect_system
        system = detect_system()
        vram = float(system.get("gpu_vram_gb") or 0)
        if vram > 0:
            return max(8, int(round(vram)))
    except Exception as exc:
        logger.debug("auto_stack vram detect failed: %s", exc)
    return 16


def _desired_stack(vram_gb: int, quant: str) -> list[str]:
    router = load_local_llm_router()
    profile = router.profile_for_vram_gb(vram_gb)
    override = get_setting("auto_stack_models", []) or []
    if isinstance(override, list) and override:
        return [str(m).strip() for m in override if str(m).strip()]
    return list(router.recommended_models(profile, quant=quant))


def build_model_pool(
    endpoint_url: str,
    *,
    owner: str | None = None,
) -> list[str]:
    check_auto_stack_ready(endpoint_url, owner=owner)
    installed = installed_tags_for_endpoint(endpoint_url, owner=owner)
    vram_gb = _detect_vram_gb()
    quant = str(get_setting("auto_stack_quant", "qat") or "qat").strip() or "qat"
    override = get_setting("auto_stack_models", []) or []
    if isinstance(override, list) and override:
        pool = [str(m).strip() for m in override if str(m).strip() in installed]
        if len(pool) >= 2:
            return pool
    desired = _desired_stack(vram_gb, quant)
    pool = [m for m in desired if m in installed]
    if len(pool) < 2:
        # Route only across models that are actually on the endpoint — never
        # invent preset tags (gemma4:e4b, etc.) that are not installed.
        pool = list(installed)
    if len(pool) < 2:
        raise AutoStackNotReady(
            f"Auto (Local LLMs) needs 2+ installed models on your local endpoint; found {len(pool)}. "
            "Open Cookbook to add models, then refresh endpoints.",
            code="insufficient_models",
        )
    return pool


def resolve_auto_stack(
    *,
    prompt: str,
    endpoint_url: str,
    headers: dict | None,
    owner: str | None = None,
    mode: str,
) -> AutoStackResolution:
    router = load_local_llm_router()
    pool = build_model_pool(endpoint_url, owner=owner)
    vram_gb = _detect_vram_gb()
    quant = str(get_setting("auto_stack_quant", "qat") or "qat").strip() or "qat"
    router.configure(vram_gb=vram_gb, quant=quant, models=pool)
    decision = router.explain(prompt, mode=mode)
    tier = decision.tier
    tag = decision.model
    reasons = tuple(decision.reasons)
    url, model, hdrs = resolve_model_on_endpoint(
        tag,
        endpoint_url=endpoint_url,
        headers=headers,
        owner=owner,
    )
    logger.info(
        "[auto_stack] tier=%s model=%s mode=%s reasons=%s pool=%s",
        tier,
        model,
        mode,
        "; ".join(reasons),
        ",".join(pool),
    )
    return AutoStackResolution(
        endpoint_url=url,
        model=model,
        headers=hdrs,
        tier=str(getattr(tier, "value", tier)),
        route_reasons=reasons,
        pool=tuple(pool),
    )


def stack_fallback_candidates(
    resolution: AutoStackResolution,
    *,
    endpoint_url: str,
    headers: dict | None,
    owner: str | None = None,
) -> list[tuple[str, str, dict]]:
    """Other models in the stack pool on the same endpoint (no cloud fallbacks)."""
    out: list[tuple[str, str, dict]] = []
    for tag in resolution.pool:
        if tag == resolution.model:
            continue
        try:
            url, model, hdrs = resolve_model_on_endpoint(
                tag,
                endpoint_url=endpoint_url,
                headers=headers,
                owner=owner,
            )
            out.append((url, model, hdrs))
        except ValueError:
            continue
    return out
