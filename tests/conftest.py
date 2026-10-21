import pytest

from app.cache import get_cache


@pytest.fixture(autouse=True)
def _clear_semantic_cache():
    get_cache().clear()
    yield
    get_cache().clear()
