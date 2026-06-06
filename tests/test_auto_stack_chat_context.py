from types import SimpleNamespace

from src.auto_stack_router import is_auto_stack_model


def test_auto_stack_skips_normalization_path():
    sess = SimpleNamespace(model="__auto_stack__", endpoint_url="http://127.0.0.1:11434")
    assert is_auto_stack_model(sess.model)
