import pytest

from draf.prep import get_el_SLP
from draf.time_series_analyzer import DemandAnalyzer


@pytest.mark.slow
def test_tsa():
    year, freq = 2020, "60min"
    p_el = get_el_SLP(year, freq, annual_energy=10e6, profile="G0")

    da = DemandAnalyzer(p_el, year, freq)
    da.show_stats()

    pla = da.get_peak_load_analyzer()
    pla.histo(target_percentile=95)
    pla.simulate_BES(e_bes_capa=2.0, p_bes_max=2.0)
