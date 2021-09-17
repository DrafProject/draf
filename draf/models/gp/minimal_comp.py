"""A small script using the model pv_bes."""

import draf
from draf.model_builder.components import *


def main():
    cs = draf.CaseStudy(name="min_imp", year=2019, freq="60min", country="DE", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(doc="no BES", components=[PRE, eDem, EG(c_buyPeak=50), PV, POST])
    cs.add_scens(nParetoPoints=2)
    cs.optimize()
    return cs
