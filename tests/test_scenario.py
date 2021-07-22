import pytest
from draf import CaseStudy, Scenario


@pytest.fixture
def case() -> CaseStudy:
    return CaseStudy()


@pytest.fixture
def scen(case) -> Scenario:
    scen = Scenario(cs=case)
    case.scens = scen
    return scen


@pytest.fixture
def scen(case) -> Scenario:
    scen = Scenario(cs=case)
    case.scens = scen
    return scen


def test___init__(case, scen):
    assert hasattr(scen, "id")
    assert case.scens == scen


def test_update_par_dic(scen):
    scen.update_par_dic()
    assert scen._par_dic == {}


@pytest.mark.parametrize("what, expected", [["params", ""], ["res", "Scenario has no res_dic."]])
def test_get_var_par_dic(what: str, expected: str, scen):
    if what is "params":
        assert scen.get_var_par_dic(what=what) == dict(expected)
    else:
        with pytest.raises(RuntimeError) as exc_info:
            scen.get_var_par_dic(what=what)
            assert exc_info == expected


def test_activate_vars(scen):
    assert isinstance(scen._activate_vars(), Scenario)
    assert scen._activate_vars().year == 2019


def test__all_ents_dict(scen):
    assert scen._all_ents_dict == {"_changed_since_last_dic_export": False}


def test_par_dic(scen):
    assert scen.par_dic == {}
