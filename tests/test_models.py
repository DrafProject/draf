import pytest

from draf.models.gp import der_hut, der_hut_comp, minimal, minimal_import, pv, pv_bes, pv_bes_comp
from draf.models.pyo import pv as pyo_pv


def test_pyo():
    assert pv.main().REF_scen.res.C_TOT_ == pytest.approx(pyo_pv.main().REF_scen.res.C_TOT_)


def test_comp():
    assert pv_bes.main().REF_scen.res.C_TOT_ == pytest.approx(
        pv_bes_comp.main().REF_scen.res.C_TOT_
    )


@pytest.mark.slow
@pytest.mark.parametrize("mdl", [minimal, minimal_import, der_hut, der_hut_comp])
def test_models(mdl):
    c = mdl.main().REF_scen.res.C_TOT_
    assert isinstance(c, float)
