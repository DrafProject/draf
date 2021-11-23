import pytest

from draf.models.gp import (
    bev_comp,
    der_hut_comp,
    minimal,
    minimal_comp,
    prod_comp,
    pv,
    pv_bes,
    pv_bes_comp,
)
from draf.models.pyo import pv as pyo_pv


@pytest.mark.slow
def test_pyo():
    assert pv.main().REF_scen.res.C_TOT_ == pytest.approx(pyo_pv.main().REF_scen.res.C_TOT_)


@pytest.mark.gurobi
def test_comp():
    assert pv_bes.main().REF_scen.res.C_TOT_ == pytest.approx(
        pv_bes_comp.main().REF_scen.res.C_TOT_
    )


@pytest.mark.gurobi
@pytest.mark.parametrize("mdl", [minimal, minimal_comp, der_hut_comp, bev_comp, prod_comp])
def test_models(mdl):
    c = mdl.main().REF_scen.res.C_TOT_
    assert isinstance(c, float)
