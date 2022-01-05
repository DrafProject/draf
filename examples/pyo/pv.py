"""This minimal example simulates the electricity purchase for a given demand and PV production.
All variables are solves in the presolve. The aim of this file is to show the syntax of parameter
& model definition. They are used by the main function to create a case-study with 2 scenarios: 
`REF` and `REF_PV`.
"""

import pyomo.environ as pyo
from gurobipy import GRB

import draf
from draf import Collectors, Dimensions, Params, Results, Scenario, Vars
from draf.abstract_component import Component

class TEST_COMP(Component):
    
    def param_func(_, sc: draf.Scenario):

        # General
        sc.dim("T", infer=True)
        sc.prep.k__dT_()

        # Total
        sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

        # Pareto
        sc.param("k_PTO_alpha_", 0, "Pareto weighting factor", "")

        # EG
        sc.prep.c_EG_RTP_T()
        sc.prep.ce_EG_T()
        sc.var("P_EG_buy_T", doc="Purchasing electrical power", unit="kW_el", lb=-GRB.INFINITY)

        # Demand
        sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)

        # PV
        sc.param("P_PV_CAPx_", 0, "existing capacity", "kW_peak")
        sc.prep.P_PV_profile_T(use_coords=True)


    def model_func(_, sc: Scenario, m: pyo.Model, d: Dimensions, p: Params, v: Vars, c: Collectors):

        m.obj = pyo.Objective(
            expr=(1 - p.k_PTO_alpha_) * v.C_TOT_ + p.k_PTO_alpha_ * v.CE_TOT_, sense=pyo.minimize
        )

        # C
        m.DEF_C_ = pyo.Constraint(
            expr=(
                v.C_TOT_
                == p.k__dT_ * pyo.quicksum(v.P_EG_buy_T[t] * p.c_EG_RTP_T[t] / 1e3 for t in d.T)
            )
        )

        # CE
        m.DEF_CE_ = pyo.Constraint(
            expr=(v.CE_TOT_ == p.k__dT_ * pyo.quicksum(v.P_EG_buy_T[t] * p.ce_EG_T[t] for t in d.T))
        )

        # Electricity
        m.BAL_pur = pyo.Constraint(
            d.T,
            rule=lambda v, t: v.P_EG_buy_T[t] + p.P_PV_CAPx_ * p.P_PV_profile_T[t] == p.P_eDem_T[t],
        )


def main():
    cs = draf.CaseStudy("pv_pyo", year=2019, freq="60min", coords=(49.01, 8.39), mdl_language="pyo")
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(components=[TEST_COMP])
    cs.add_scen("REF_PV", doc="REF plus PV").update_params(P_PV_CAPx_=100)
    cs.optimize(which_solver="glpk")
    return cs
