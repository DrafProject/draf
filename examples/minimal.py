"""Minimal example with only one user-defined component.

Notes:
    - The model just determines the total costs `C_TOT_`. There is nothing to "optimize".
    - param_func is DRAF syntax, the model_func is GurobiPy syntax.
    - Most of the CaseStudy functions can be chained.
"""

from gurobipy import GRB, Model, quicksum

import draf
from draf import Collectors, Dimensions, Params, Scenario, Vars  # only used for type hinting
from draf.abstract_component import Component  # only used for type hinting


class Minimal(Component):
    def param_func(self, sc: Scenario):

        # Define the optimization variable C_TOT_:
        sc.var("C_TOT_", doc="Total costs", unit="€/a")

        # Prepare time-dependent day-ahead market prices as parameter c_EG_RTP_T:
        sc.prep.c_EG_RTP_KG()

        # Prepare a time-dependent G1 standard load profile (Business on weekdays 08:00 - 18:00)
        # with the annual energy of 5 GWh:
        sc.prep.P_eDem_KG(profile="G1", annual_energy=5e6)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):

        # Set the objective function:
        m.setObjective(v.C_TOT_, GRB.MINIMIZE)

        # Add a constraint to the model
        m.addConstr(
            v.C_TOT_
            == quicksum(
                p.P_eDem_KG[k, g] * sc.dt(k, g) * sc.periodOccurrences[k] * p.c_EG_RTP_KG[k, g]
                for k in d.K
                for g in d.G
            )
        )


def main():
    cs = draf.CaseStudy()
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen(components=[Minimal])
    cs.optimize()
    return cs
