import pytest

from examples import bev, der_hut, minimal, pv, pv_bes, pv_bes_TSA, pyomo_pv


@pytest.mark.gurobi
@pytest.mark.parametrize("mdl", [bev, der_hut, minimal, pv, pv_bes, pv_bes_TSA, pyomo_pv])
def test_examples(mdl):
    c = mdl.main().REF_scen.res.C_TOT_
    assert isinstance(c, float)
