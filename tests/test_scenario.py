import pandas as pd
import pytest

from draf import CaseStudy, Scenario


@pytest.fixture
def cs() -> CaseStudy:
    return CaseStudy()


@pytest.fixture
def sc(cs) -> Scenario:
    sc = Scenario(cs=cs)
    cs.scens = sc
    return sc


@pytest.fixture
def sc(cs) -> Scenario:
    sc = Scenario(cs=cs)
    cs.scens = sc
    return sc


@pytest.mark.parametrize("what, expected", [["params", ""], ["res", "Scenario has no res_dic."]])
def test_get_var_par_dic(what: str, expected: str, sc):
    if what is "params":
        assert sc.get_var_par_dic(what=what) == dict(expected)
    else:
        with pytest.raises(RuntimeError) as exc_info:
            sc.get_var_par_dic(what=what)
            assert exc_info == expected


def test_activate_vars(sc):
    assert isinstance(sc._activate_vars(), Scenario)
    assert sc._activate_vars().year == 2019


def test_trim_to_datetimeindex(cs):
    cs.set_datetime_filter(start="Jan-02 00", steps=24 * 2)
    sc = cs.add_REF_scen()
    ser = pd.Series(index=range(8760))
    df = ser.to_frame()
    for input in (ser, df):
        new = sc.trim_to_datetimeindex(input)
        assert new.index[0] == 24
        assert new.index[-1] == 71
