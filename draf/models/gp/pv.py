"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios:
`REF` and `REF_PV`.
"""

from gurobipy import GRB, Model, quicksum

import draf


def params_func(sc: draf.Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.var("C_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("alpha_", data=0, doc="Pareto weighting factor")

    # GRID
    sc.prep.c_GRID_RTP_T()
    sc.prep.ce_GRID_T()
    sc.var("E_GRID_buy_T", doc="Purchased electricity", unit="kWh_el", lb=-GRB.INFINITY)

    # Demand
    sc.prep.E_dem_T(profile="G1", annual_energy=5e6)

    # PV
    sc.param("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
    sc.prep.E_PV_profile_T(use_coords=True)


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):

    m.setObjective((1 - p.alpha_) * v.C_ + p.alpha_ * v.CE_, GRB.MINIMIZE)

    # C
    m.addConstr(
        v.C_ == quicksum(v.E_GRID_buy_T[t] * p.c_GRID_RTP_T[t] / 1e3 for t in d.T), "DEF_C_"
    )

    # CE
    m.addConstr(v.CE_ == quicksum(v.E_GRID_buy_T[t] * p.ce_GRID_T[t] for t in d.T), "DEF_CE_")

    # Electricity
    m.addConstrs(
        (v.E_GRID_buy_T[t] + p.P_PV_CAPx_ * p.E_PV_profile_T[t] == p.E_dem_T[t] for t in d.T),
        "BAL_pur",
    )


def main():
    cs = draf.CaseStudy("pv_gp", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.set_model(model_func).optimize()
    return cs
