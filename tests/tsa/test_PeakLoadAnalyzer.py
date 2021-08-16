import pytest

from draf import paths
from draf.helper import read
from draf.tsa import PeakLoadAnalyzer


@pytest.mark.slow
def test_histo():
    ser = read(paths.DATA / "demand/electricity/test_G1.parquet")
    pla = PeakLoadAnalyzer(p_el=ser.values)
    pla.histo(peak_reduction=0.01 * ser.max())
    pla.simulate_BES(bes_capa=2.0, bes_P_max=2.0)
