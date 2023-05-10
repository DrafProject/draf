"""Example model for Distributed Energy Resources (DER) and Heat Upgrading Technologies (HUT)"""


import draf
from draf.components import *


def main():
    cs = draf.CaseStudy(
        "der_hut", year=2019, freq="60min", coords=(49.01, 8.39), consider_invest=True
    )
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    sc = cs.add_REF_scen(components=[cDem, eDem, hDem, BES, CHP, EG, Fuel, HD,
                                     HOB, HP, P2H, PV, TES, Main])  # fmt: skip
    cs.optimize(logToConsole=False)
    return cs
