"""Optimizing the operation of a 100 kWh Battery Energy Storage (BES) considering a Photovoltaic
(PV), a fixed electricity demand (eDem), and an existing 1 MWp Photovoltaic system.

The example is used in the DRAF showcase see: https://mfleschutz.github.io/draf-showcase/#/2
"""

import draf
from draf.components import *


def main():
    cs = draf.CaseStudy("pv_bes", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Aug-01 00:00", steps=24)
    sc = cs.add_REF_scen(components=[eDem, BES(E_CAPx=100), PV(P_CAPx=1e3), EG, Main])
    cs.add_scens([("c_EG_T", "t", ["c_EG_TOU_T", "c_EG_FLAT_T"])])
    cs.optimize(logToConsole=False)
    return cs
