import collections
import time

import draf
import pandas as pd
import pytest


@pytest.fixture
def case() -> draf.CaseStudy:
    return draf.CaseStudy()


def test___init__(case):
    assert case.name == "test"
    if case.freq not in ["15min", "60min"]:
        with pytest.raises(AssertionError):
            draf.__init__()


set_year_con = [[1980], [2020], [2101]]


@pytest.mark.parametrize("year", set_year_con)
def test_set_year(year: int, case):
    if year in range(1980, 2100):
        case.set_year(year=year)
        assert case.year == year
    else:
        with pytest.raises(AssertionError):
            case.set_year(year=year)


step_width_con = [["15min", 0.25], ["30min", 0.5], ["60min", 1.0]]


@pytest.mark.parametrize("freq, expected", step_width_con)
def test_step_width(freq: str, expected: float, case):
    case.freq = freq
    assert case.step_width == expected


def test_get_T(case):
    t_list = []
    for t in range(8760):
        t_list.append(t)
    assert case.get_T() == t_list


_set_dtindex_container = [["15min", 96, "1/4 h"], ["30min", 24, "h"], ["60min", 24, "h"]]


@pytest.mark.parametrize("freq, steps, unit", _set_dtindex_container)
def test__set_dtindex(freq: str, steps: int, unit: str, case):
    case.freq = freq
    case._set_dtindex()
    assert case.steps_per_day == steps
    assert case._freq_unit == unit


def test__set_time_trace(case):
    case._set_time_trace()
    expected = time.time()
    assert case._time == expected


def test__get_time_diff(case):
    case._set_time_trace()
    assert time.time() - case._time == case._get_time_diff()


# set_datetime_filter(), _get_integer_locations() and _get_datetime_int_loc_from_string() are also tested
set_time_horizon_container = [
    ["May1 00:00", None, "Jun1 23:00", 2880, 3647],
    ["May1 00:00", 24 * 30, None, 2880, 3599],
    ["Oct3 20:00", None, None, 6620, 8759],
    ["Dec15 00:00", None, "Dec15 15:00", 8352, 8367],
]


@pytest.mark.parametrize("start, steps, end, t1, t2", set_time_horizon_container)
def test_set_time_horizon(start: str, steps: str, end: str, t1: int, t2: int, case):
    case.set_time_horizon(start=start, steps=steps, end=end)
    assert case._t1 == t1
    assert case._t2 == t2


get_datetime_int_loc_from_string_container = [
    ["1", slice(0, 744, None)],
    ["2", slice(744, 1416, None)],
    ["10", slice(6552, 7296, None)],
]


@pytest.mark.parametrize("string, expected", get_datetime_int_loc_from_string_container)
def test__get_datetime_int_loc_from_string(string: str, expected: str, case):
    assert case._get_datetime_int_loc_from_string(s=string) == expected


@pytest.fixture
def case() -> draf.CaseStudy:
    return draf.CaseStudy()


def test_scens_list(case):
    assert case.scens_list == []


def test_scens_dic(case):
    assert case.scens_dic == {}


# def test_any_scen:
# any_scen returns a CaseStudy object, although it should return a Scenario object


def test_valid_scens(case):
    assert case.valid_scens == {}


def test_ordered_valid_scens(case):
    assert case.ordered_valid_scens == collections.OrderedDict()


def test_pareto(case):
    assert isinstance(case.pareto, pd.DataFrame)


# def test_REF_scen:
# REF_scen returns a CaseStudy object, although it should return a Scenario object


def test_dt_info(case):
    assert isinstance(case.dt_info, str)
