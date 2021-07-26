"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios."""

from gurobipy import GRB, quicksum

import draf


def params_func(sc):
    sc.add_dim("T", infer=True)
    sc.add_par("alpha_", 0, "weighting factor", "")
    sc.add_par("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
    sc.prep.add_c_GRID_RTP_T()
    sc.prep.add_ce_GRID_T()
    sc.prep.add_E_PV_profile_T()
    sc.prep.add_E_dem_T(profile="G1", annual_energy=4.8e6)

    sc.add_var("C_", "total costs", "k€/a", lb=-GRB.INFINITY)
    sc.add_var("CE_", "total emissions", "kgCO2eq/a", lb=-GRB.INFINITY)
    sc.add_var("E_pur_T", "purchased electricity", "kWh_el", lb=-GRB.INFINITY)


def model_func(m, d, p, v):
    m.setObjective((1 - p.alpha_) * v.C_ + p.alpha_ * v.CE_, GRB.MINIMIZE)
    m.addConstr(v.C_ == quicksum(v.E_pur_T[t] * p.c_GRID_RTP_T[t] / 1e3 for t in d.T), "DEF_C_")
    m.addConstr(v.CE_ == quicksum(v.E_pur_T[t] * p.ce_GRID_T[t] for t in d.T), "DEF_CE_")
    m.addConstrs((v.E_pur_T[t] + p.P_PV_CAPx_ * p.E_PV_profile_T[t] == p.E_dem_T[t] for t in d.T), "BAL_pur")


def main():
    cs = draf.CaseStudy("minimal", year=2019, freq="60min")
    cs.add_REF_scen().set_params(params_func)
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.set_model(model_func).optimize()
    return cs


if __name__ == "__main__":
    cs = main()
