import pandas as pd

from draf import helper as hp


def test_datetime_to_int():
    assert hp.datetime_to_int(freq="60min", year=2019, month=12, day=31) == 8760 - 24


def test_int_to_datetime():
    assert isinstance(hp.int_to_datetime(freq="60min", year=2019, pos=8760 - 24), pd.Timestamp)


def test_auto_fmt():
    num, unit = hp.auto_fmt(num=2e4, unit="kW")
    assert (num, unit) == (20.0, "MW")

    num, unit = hp.auto_fmt(num=2e4, unit="€")
    assert (num, unit) == (20.0, "k€")

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


def test_sizeof_fmt():
    assert hp.sizeof_fmt(num=1400, suffix="B") == "  1.4 KB"