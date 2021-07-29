import pytest

from draf.models.gp import der_hut, investment, minimal, mo_bes, mo_bes_2, p1
from draf.models.pyo import minimal as pyo_minimal


@pytest.mark.parametrize("mdl", [minimal, pyo_minimal])
def test_minimal(mdl):
    mdl.main()


@pytest.mark.slow
@pytest.mark.parametrize("mdl", [der_hut, investment, mo_bes, mo_bes_2, p1])
def test_models(mdl):
    mdl.main()
