import pandas as pd
import pytest

from draf import helper as hp


def test_get_etype():
    assert hp.get_etype("c_") == "c"
    assert hp.get_etype("c_comp") == "c"
    assert hp.get_etype("c_comp_DIMS") == "c"
    assert hp.get_etype("c_comp_desc_DIMS") == "c"


def test_get_dims():
    assert hp.get_component("c_") == ""
    assert hp.get_component("c_DIMS") == "DIMS"
    assert hp.get_component("c_COMP_DIMS") == "DIMS"
    assert hp.get_component("c_COMP_desc_DIMS") == "DIMS"


def test_get_desc():
    assert hp.get_desc("c_") == ""
    assert hp.get_desc("c_DIMS") == ""
    assert hp.get_desc("c_COMP_DIMS") == ""
    assert hp.get_desc("c_COMP_desc_DIMS") == "desc"
    assert hp.get_desc("c_COMP_desc_addon_DIMS") == "desc"
    assert hp.get_desc("c__desc_") == "desc"


def test_get_dims():
    assert hp.get_dims("x_") == ""
    assert hp.get_dims("x_DIMS") == "DIMS"
    assert hp.get_dims("x_comp_DIMS") == "DIMS"
    assert hp.get_dims("x_comp_desc_DIMS") == "DIMS"


def test_datetime_to_int():
    assert hp.datetime_to_int(freq="60min", year=2019, month=12, day=31) == 8760 - 24


def test_int_to_datetime():
    assert isinstance(hp.int_to_datetime(freq="60min", year=2019, pos=8760 - 24), pd.Timestamp)


def test_auto_fmt():
    num, unit = hp.auto_fmt(num=2e4, unit="kW")
    assert (num, unit) == (20.0, "MW")

    num, unit = hp.auto_fmt(num=2e6, unit="€")
    assert (num, unit) == (2.0, "M€")

    num, unit = hp.auto_fmt(num=2e6, unit="€", target_unit="k€")
    assert (num, unit) == (2000.0, "k€")

    num, unit = hp.auto_fmt(num=2e4, unit="gCO2eq")
    assert (num, unit) == (20.0, "kgCO2eq")


def test_wrap_and_border():
    assert hp.wrap_and_border("spam and eggs", 6) == "┌────┐\n│spam│\n│and │\n│eggs│\n└────┘"


def test_bordered():
    assert hp.bordered("spam") == "┌────┐\n│spam│\n└────┘"


def test_ser_to_df_for_latex_table():
    ser = pd.Series(dict(a=3, b=2, c=4))
    result = hp.ser_to_df_for_latex_table(ser, ncols=2)
    assert result.__repr__() == "  index  0 index  0\n0     a  3     b  2"


def test_human_readable_size():
    assert hp.human_readable_size(size=1400, decimal_places=2) == "1.40 KB"


def test_topological_sort():
    order_restriction_with_cyclic_dependency = [
        ("A", {"B"}),
        ("B", {"C"}),
        ("C", {"A"}),
    ]
    with pytest.raises(ValueError):
        list(hp.topological_sort(order_restriction_with_cyclic_dependency))
