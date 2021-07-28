import draf
import gurobipy as gp


def test_make_minimal_model():
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


def test_parameterize_existing_model():
    from draf.models import pv_bes as mod

    cs = draf.CaseStudy(name="ShowCase", year=2017, freq="60min", country="DE")
    cs.set_time_horizon(start=0, steps=24)

    cs.add_REF_scen(doc="no BES").set_params(mod.params_func).update_params(
        P_PV_CAPx_=100, c_GRID_peak_=50
    )

    cs.add_scens(
        scen_vars=[
            ("c_GRID_T", "t", ["c_GRID_RTP_T", "c_GRID_TOU_T"]),
            ("E_BES_CAPx_", "b", [1000]),
        ],
        nParetoPoints=4,
    )

    cs.improve_pareto_and_set_model(mod.model_func).optimize(postprocess_func=mod.postprocess_func)
