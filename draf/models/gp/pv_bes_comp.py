"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

import draf
from draf.model_builder.components import BES, PV, Base


def main():
    cs = draf.CaseStudy("pv_bes_comp", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(components=[Base, BES, PV]).update_params(P_PV_CAPx_=100, E_BES_CAPx_=100)
    cs.add_scens([("c_EL_T", "t", ["c_EL_TOU_T", "c_EL_FLAT_T"])])
    cs.optimize(logToConsole=False)
    return cs
