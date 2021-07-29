import pandas as pd
import pytest

from draf import prep


@pytest.mark.slow
def test_get_el_SLP():
    ser = prep.get_el_SLP(year=2019, country="DE", freq="60min", profile="G1", annual_energy=1e6)
    assert isinstance(ser, pd.Series)
    assert ser.sum() == pytest.approx(1e6)

    ser = prep.get_el_SLP(
        year=2019, country="DE", freq="60min", profile="G1", offset=1e3, peak_load=5e3
    )
    assert isinstance(ser, pd.Series)
    assert ser.max() == pytest.approx(5e3)


def test_get_thermal_demand():
    t_amb = pd.Series(12, index=range(8760))
    ser = prep.get_thermal_demand(ser_amb_temp=t_amb, year=2019)
    assert isinstance(ser, pd.Series)
    assert ser.min() > 0


def test_get_cooling_demand():
    t_amb = pd.Series(30, index=range(8760))
    t_amb[2] = 15
    ser = prep.get_cooling_demand(ser_amb_temp=t_amb, year=2019)
    assert isinstance(ser, pd.Series)
    assert ser.min() >= 0.0
    assert ser[2] == 0.0
