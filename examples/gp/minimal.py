"""A very short model that populates the variables and parameters without `params_func`"""

from gurobipy import GRB, Model, quicksum

import draf
from draf import Collectors, Dimensions, Params, Scenario, Vars
from draf.abstract_component import Component


class Minimal(Component):
    def param_func(self, sc: Scenario):
        sc.dim("T", infer=True)
        sc.prep.k__dT_()
        sc.var("C_TOT_", unit="€/a")
        sc.prep.c_EG_RTP_T()
        sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.setObjective(v.C_TOT_, GRB.MINIMIZE)
        m.addConstr(v.C_TOT_ == p.k__dT_ * quicksum(p.P_eDem_T[t] * p.c_EG_RTP_T[t] for t in d.T))


def main():
    cs = draf.CaseStudy()
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(components=[Minimal]).optimize()
    return cs
