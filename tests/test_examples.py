import pytest

from examples import bev, der_hut, minimal, prod, pv, pv_bes, pyomo_pv


@pytest.mark.gurobi
@pytest.mark.parametrize("mdl", [bev, der_hut, minimal, prod, pv, pv_bes, pyomo_pv])
def test_examples(mdl):
    c = mdl.main().REF_scen.res.C_TOT_
    assert isinstance(c, float)
