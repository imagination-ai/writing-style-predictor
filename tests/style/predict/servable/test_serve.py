import pytest

from style.predict.servable.base import MockServable
from style.predict.servable.base import SklearnBasedClassifierServable
from style.predict.servable.serve import get_servable


@pytest.mark.parametrize(
    "case, result",
    [("mock", MockServable), ("small", SklearnBasedClassifierServable)],
)
def test_get_servable(case, result):
    assert isinstance(get_servable(case), result)
