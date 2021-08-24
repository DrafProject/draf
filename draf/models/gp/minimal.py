"""A very short model that populates the variables and parameters without `params_func`"""

from gurobipy import GRB, Model, quicksum

import draf


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    m.setObjective(v.C_TOT_, GRB.MINIMIZE)
    m.addConstr(v.C_TOT_ == p.k__dT_ * quicksum(p.P_eDem_T[t] * p.c_GRID_RTP_T[t] for t in d.T))


def main():
    cs = draf.CaseStudy()
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)

    sc = cs.add_REF_scen()
    sc.dim("T", infer=True)
    sc.prep.k__dT_()
    sc.var("C_TOT_", unit="€/a")
    sc.prep.c_GRID_RTP_T()
    sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)

    cs = cs.set_model(model_func).optimize()
    return cs
