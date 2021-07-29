import pytest

from draf import prep


@pytest.mark.slow
def test_get_pv_power():
    ser = prep.get_pv_power(year=2019, coords=(49.01, 8.39))
    assert ser.max() == pytest.approx(0.837, rel=1e-3)
