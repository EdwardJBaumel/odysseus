from src.auto_stack_router import is_auto_stack_model, is_auto_stack_session
from src.constants import AUTO_STACK_MODEL_ID


def test_auto_stack_model_constant():
    assert AUTO_STACK_MODEL_ID == "__auto_stack__"
    assert is_auto_stack_model(AUTO_STACK_MODEL_ID)


def test_auto_stack_session_by_model_only():
    class Sess:
        model = AUTO_STACK_MODEL_ID

    assert is_auto_stack_session(Sess()) is True
    assert is_auto_stack_session(Sess(), require_enabled=True) is True
