import gurobipy as gp

import draf


def main():

    cs = draf.CaseStudy(name="foo", year=2019, freq="60min", country="DE")
    cs.set_time_horizon(start=0, steps=24)

    sc = cs.add_REF_scen()
    sc.add_dim("T", infer=True)
    sc.prep.add_c_GRID_RTP_T()
    sc.prep.add_E_dem_T(profile="G3", annual_energy=5e5)
    sc.add_var("C_", unit="€/a", lb=-gp.GRB.INFINITY)

    def model_func(d, p, v, m):  # (d)imensions, (p)arameters, (v)ariables, (m)odel
        m.setObjective(v.C_, gp.GRB.MINIMIZE)
        m.addConstr(v.C_ == gp.quicksum(p.E_dem_T[t] * p.c_GRID_RTP_T[t] for t in d.T))

    cs.set_model(model_func).optimize()


if __name__ == "__main__":
    cs = main()
