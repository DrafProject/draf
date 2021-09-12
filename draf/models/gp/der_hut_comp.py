"""DER-HUT: Distributed Energy Resources - Heat Upgrading Technologies"""


import draf
from draf.model_builder.components import *


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(components=[PRE, cDem, eDem, hDem, BES, CHP, EG, Fuel, H2H1,
                                HOB, HP, P2H, PV, TES, POST])  # fmt: skip
    cs.optimize(logToConsole=False)
    return cs
