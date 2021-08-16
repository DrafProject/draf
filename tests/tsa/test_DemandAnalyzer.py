import pytest

from draf import paths
from draf.helper import read
from draf.tsa import DemandAnalyzer


@pytest.mark.slow
def test_histo():
    ser = read(paths.DATA / "demand/electricity/test_G1.parquet")
    da = DemandAnalyzer(p_el=ser, year=2019, freq="15min")
    da.show_stats()
