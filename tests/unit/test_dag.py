import pytest

from src.dag import DAG
from src.operators import Source


def test_source_without_value_raises() -> None:
    with pytest.raises(ValueError, match="source node needs a value"):
        DAG(Source())  # type: ignore[call-overload]
