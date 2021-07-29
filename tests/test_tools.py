import pytest

from draf import paths
from draf.helper import read
from draf.tools import PeakLoadAnalysis


@pytest.mark.slow
def test_histo():
    ser = read(paths.DATA / "demand/electricity/test_G1.parquet")
    pla = PeakLoadAnalysis(p_el=ser.values)
    pla.histo(peak_reduction=0.01 * ser.max())
    pla.simulate(bes_capa=2.0, bes_P_max=2.0)
