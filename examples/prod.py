"""Flexible operation of an industrial process"""


import draf
from draf.components import *


def main():
    cs = draf.CaseStudy(
        "prod", year=2019, freq="60min", coords=(49.01, 8.39), consider_invest=False
    )
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    sc = cs.add_REF_scen(components=[eDem, EG, PP, PS, PROD, Main])  # fmt: skip
    cs.optimize(logToConsole=False)
    return cs
