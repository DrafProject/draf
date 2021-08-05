"""A very short model without params_func"""


def main():
    from gurobipy import GRB, Model, quicksum

    import draf

    cs = draf.CaseStudy(name="foo", year=2019, freq="60min", country="DE", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)

    sc = cs.add_REF_scen()
    sc.dim("T", infer=True)
    sc.var("C_", unit="€/a", lb=-GRB.INFINITY)
    sc.prep.c_GRID_RTP_T()
    sc.prep.E_dem_T(profile="G3", annual_energy=5e6)

    def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
        m.setObjective(v.C_, GRB.MINIMIZE)
        m.addConstr(v.C_ == quicksum(p.E_dem_T[t] * p.c_GRID_RTP_T[t] for t in d.T))

    cs.set_model(model_func).optimize()
    return cs
