"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios."""

import pyomo.environ as pyo
from gurobipy import GRB

from draf import CaseStudy, Dimensions, Params, Scenario, Vars


def params_func(sc: Scenario):
    sc.add_dim("T", infer=True)
    sc.add_par("alpha_", 0, "weighting factor", "")
    sc.add_par("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
    sc.prep.add_c_GRID_RTP_T()
    sc.prep.add_ce_GRID_T()
    sc.prep.add_E_PV_profile_T()
    sc.prep.add_E_dem_T(profile="G1", annual_energy=4.8e6)

    sc.add_var("C_", doc="total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.add_var("CE_", doc="total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)
    sc.add_var("E_pur_T", doc="purchased electricity", unit="kWh_el", lb=-GRB.INFINITY)


def model_func(m: pyo.Model, d: Dimensions, p: Params, v: Vars):

    m.obj = pyo.Objective(expr=(1 - p.alpha_) * v.C_ + p.alpha_ * v.CE_, sense=pyo.minimize)
    m.DEF_C_ = pyo.Constraint(
        expr=(v.C_ == pyo.quicksum(v.E_pur_T[t] * p.c_GRID_RTP_T[t] / 1e3 for t in d.T))
    )
    m.DEF_CE_ = pyo.Constraint(
        expr=(v.CE_ == pyo.quicksum(v.E_pur_T[t] * p.ce_GRID_T[t] for t in d.T))
    )

    m.BAL_pur = pyo.Constraint(
        d.T, rule=lambda v, t: v.E_pur_T[t] + p.P_PV_CAPx_ * p.E_PV_profile_T[t] == p.E_dem_T[t]
    )


def main():
    cs = CaseStudy("minimal", year=2019, freq="60min")
    cs.set_datetime_filter(start="Apr-01 00", steps=24 * 10)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.add_scens(nParetoPoints=2, based_on="REF_PV")
    cs.set_model(model_func, mdl_language="pyo").optimize(which_solver="glpk")
    return cs


if __name__ == "__main__":
    cs = main()
