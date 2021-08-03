import pytest

from draf.models.gp import der_hut, minimal, minimal_import, pv, pv_bes, pv_bes_mo
from draf.models.pyo import pv as pyo_pv


@pytest.mark.parametrize("mdl", [pv, pyo_pv])
def test_pv(mdl):
    mdl.main()


@pytest.mark.slow
@pytest.mark.parametrize("mdl", [der_hut, minimal_import, minimal, pv_bes_mo, pv_bes])
def test_models(mdl):
    mdl.main()
