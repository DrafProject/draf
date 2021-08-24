"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios: 
`REF` and `REF_PV`.
"""

import pyomo.environ as pyo
from gurobipy import GRB

import draf


def params_func(sc: draf.Scenario):

    # General
    sc.dim("T", infer=True)
    sc.var("C_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)
    sc.prep.k__dT_()

    # Pareto
    sc.param("k_alpha_", 0, "Pareto weighting factor", "")

    # GRID
    sc.prep.c_GRID_RTP_T()
    sc.prep.ce_GRID_T()
    sc.var("P_GRID_buy_T", doc="Purchasing electrical power", unit="kW_el", lb=-GRB.INFINITY)

    # Demand
    sc.prep.P_dem_T(profile="G1", annual_energy=5e6)

    # PV
    sc.param("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
    sc.prep.P_PV_profile_T(use_coords=True)


def model_func(m: pyo.Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):

    m.obj = pyo.Objective(expr=(1 - p.k_alpha_) * v.C_ + p.k_alpha_ * v.CE_, sense=pyo.minimize)

    # C
    m.DEF_C_ = pyo.Constraint(
        expr=(
            v.C_
            == p.k__dT_ * pyo.quicksum(v.P_GRID_buy_T[t] * p.c_GRID_RTP_T[t] / 1e3 for t in d.T)
        )
    )

    # CE
    m.DEF_CE_ = pyo.Constraint(
        expr=(v.CE_ == p.k__dT_ * pyo.quicksum(v.P_GRID_buy_T[t] * p.ce_GRID_T[t] for t in d.T))
    )

    # Electricity
    m.BAL_pur = pyo.Constraint(
        d.T,
        rule=lambda v, t: v.P_GRID_buy_T[t] + p.P_PV_CAPx_ * p.P_PV_profile_T[t] == p.P_dem_T[t],
    )


def main():
    cs = draf.CaseStudy("pv_pyo", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.set_model(model_func, mdl_language="pyo").optimize(which_solver="glpk")
    return cs
