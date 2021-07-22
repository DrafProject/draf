import pytest
from draf.core.draf_base_class import DrafBaseClass


@pytest.fixture
def dbc() -> DrafBaseClass:
    return DrafBaseClass()


def test__get_dims(dbc):
    assert dbc._get_dims("E_BEV_TH") == "TH"
