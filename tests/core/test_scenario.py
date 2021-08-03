import pandas as pd
import pytest

from draf import CaseStudy, Scenario
from draf.prep.data_base import DataBase as db


@pytest.fixture
def cs() -> CaseStudy:
    return CaseStudy(year=2019)


@pytest.fixture
def empty_sc(cs) -> Scenario:
    sc = cs.add_REF_scen()
    return sc


@pytest.fixture
def sc(cs) -> Scenario:
    sc = cs.add_REF_scen()
    sc.param("eta_test_", data=5)
    return sc


@pytest.mark.parametrize("what, expected", [["params", ""], ["res", "Scenario has no res_dic."]])
def test_get_var_par_dic(what: str, expected: str, empty_sc):
    if what is "params":
        assert empty_sc.get_var_par_dic(what=what) == dict(expected)
    else:
        with pytest.raises(RuntimeError) as exc_info:
            empty_sc.get_var_par_dic(what=what)
            assert exc_info == expected


def test__activate_vars(sc):
    assert isinstance(sc._activate_vars(), Scenario)
    assert sc._activate_vars().year == 2019


def test_trim_to_datetimeindex(cs):
    cs.set_time_horizon(start="Jan-02 00", steps=24 * 2)
    sc = cs.add_REF_scen()
    ser = pd.Series(index=range(8760))
    df = ser.to_frame()
    for input in (ser, df):
        new = sc.trim_to_datetimeindex(input)
        assert new.index[0] == 24
        assert new.index[-1] == 71


def test_get_entity(sc):
    assert sc.get_entity("eta_test_") == 5


def test_param(sc):
    sc.param(name="x_HP_test_", data=4, doc="test doc", unit="test_unit", src="test_source")
    sc.param(from_db=db.funcs.c_CHP_inv_())
    sc.param(from_db=db.eta_HP_)
    sc.param(name="c_FUEL_other-name_", from_db=db.c_FUEL_co2_)
    for ent in ["c_CHP_inv_", "eta_HP_", "c_FUEL_other-name_"]:
        isinstance(sc.params.get(ent), float)