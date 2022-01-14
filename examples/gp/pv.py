"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios:
`REF` and `REF_PV`.
"""

from gurobipy import GRB, Model, quicksum

import draf
from draf import Collectors, Dimensions, Params, Results, Scenario, Vars


def params_func(sc: Scenario):

    # Total
    sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")

    # EG
    sc.prep.c_EG_RTP_T()
    sc.prep.ce_EG_T()
    sc.var("P_EG_buy_T", doc="Purchasing electrical power", unit="kW_el", lb=-GRB.INFINITY)

    # Demand
    sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)

    # PV
    sc.param("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
    sc.prep.P_PV_profile_T(use_coords=True)


def model_func(sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):

    m.setObjective((1 - p.k_PTO_alpha_) * v.C_TOT_ + p.k_PTO_alpha_ * v.CE_TOT_, GRB.MINIMIZE)

    # C
    m.addConstr(
        v.C_TOT_ == p.k__dT_ * quicksum(v.P_EG_buy_T[t] * p.c_EG_RTP_T[t] / 1e3 for t in d.T),
        "DEF_C_",
    )

    # CE
    m.addConstr(
        v.CE_TOT_ == p.k__dT_ * quicksum(v.P_EG_buy_T[t] * p.ce_EG_T[t] for t in d.T),
        "DEF_CE_",
    )

    # Electricity
    m.addConstrs(
        (v.P_EG_buy_T[t] + p.P_PV_CAPx_ * p.P_PV_profile_T[t] == p.P_eDem_T[t] for t in d.T),
        "BAL_pur",
    )


def main():
    cs = draf.CaseStudy("pv_gp", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.set_model(model_func).optimize()
    return cs