"""Local-LLM-Router routing dependency.

local-llm-router on PyPI (MIT) picks a model tag per prompt/hint. Lazy-imported so a
broken install does not block app startup. Shipped in requirements.txt; use the
in-app install button or ``pip install 'local-llm-router[ollama]'`` if missing.

Legacy PyPI name ``split-stack`` is still accepted until it is fully retired.
"""

from __future__ import annotations

import logging

from src.constants import LOCAL_LLM_ROUTER_NAME

logger = logging.getLogger(__name__)

LOCAL_LLM_ROUTER_PIP = "local-llm-router[ollama]"
# Back-compat aliases used by settings API and older UI strings
SPLIT_STACK_PIP = LOCAL_LLM_ROUTER_PIP
LOCAL_LLM_ROUTER_MISSING = (
    f"{LOCAL_LLM_ROUTER_NAME} is not installed. "
    f"Install with `pip install '{LOCAL_LLM_ROUTER_PIP}'` or use Install in the model picker."
)
SPLIT_STACK_MISSING = LOCAL_LLM_ROUTER_MISSING


def load_local_llm_router():
    """Return the local_llm_router module, or raise a user-facing setup hint."""
    for mod_name in ("local_llm_router", "split_stack"):
        try:
            return __import__(mod_name)
        except ImportError:
            continue
    raise RuntimeError(LOCAL_LLM_ROUTER_MISSING)


def load_split_stack():
    """Deprecated alias for :func:`load_local_llm_router`."""
    return load_local_llm_router()


def local_llm_router_available() -> bool:
    try:
        load_local_llm_router()
        return True
    except RuntimeError:
        return False


def split_stack_available() -> bool:
    """Deprecated alias for :func:`local_llm_router_available`."""
    return local_llm_router_available()
