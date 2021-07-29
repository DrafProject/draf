"""An easy model with an Battery Energy Storage."""

from gurobipy import GRB, quicksum

import draf


def params_func(sc):
    T = sc.add_dim("T", infer=True)

    sc.add_par("n_CE_", 1e4)
    sc.add_par("n_C_", 1)
    sc.add_par("alpha_", 0)

    sc.add_par("pv_capa_", 0)

    sc.add_par("k_BES_in_per_capa_", 0.5, doc="ratio loading power / capacity")
    sc.add_par("k_BES_out_per_capa_", 0.5, doc="ratio unloading power / capacity")
    sc.add_par("E_BES_capa_", 0, doc="Capacity of battery energy storage.")

    sc.prep.add_c_GRID_RTP_T()
    sc.prep.add_c_GRID_TOU_T()
    sc.prep.add_c_GRID_FLAT_T()
    sc.prep.add_ce_GRID_T()
    sc.prep.add_E_dem_T()
    sc.prep.add_E_PV_profile_T()

    sc.add_par("c_GRID_T", sc.params.c_GRID_RTP_T)

    sc.add_var("C_", lb=-GRB.INFINITY, doc="total costs", unit="k€/a")
    sc.add_var(
        "CE_", lb=-GRB.INFINITY, doc="total carbon emissions",
    )
    sc.add_var("E_pur_T", lb=-GRB.INFINITY)
    sc.add_var("E_BES_delta_pos_")
    sc.add_var("E_BES_delta_neg_")
    sc.add_var("E_BES_T")
    sc.add_var("E_BES_in_T", lb=-GRB.INFINITY)


def model_func(d, p, v, m):
    T = d.T

    m.setObjective(
        (
            (1 - p.alpha_) * v.C_ / p.n_C_
            + p.alpha_ * v.CE_ / p.n_CE_
            + v.E_BES_delta_pos_
            + v.E_BES_delta_neg_
        ),
        GRB.MINIMIZE,
    )

    m.addConstr((v.C_ == quicksum(v.E_pur_T[t] * p.c_GRID_T[t] / 1e3 for t in T)), "BAL_C_")
    m.addConstr((v.CE_ == quicksum(v.E_pur_T[t] * p.ce_GRID_T[t] for t in T)), "BAL_CE_")
    m.addConstrs(
        (
            v.E_pur_T[t] + p.E_PV_profile_T[t] * p.pv_capa_ - v.E_BES_in_T[t] == p.E_dem_T[t]
            for t in T
        ),
        "BAL_TOT",
    )
    m.addConstrs(
        (v.E_BES_T[t] == 0.9994 * v.E_BES_T[t - 1] + 0.9994 * v.E_BES_in_T[t] for t in T[1:]),
        "BAL_BES",
    )
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_capa_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_in_T[t] <= p.k_BES_in_per_capa_ * p.E_BES_capa_ for t in T), "MAX_BES_IN")
    m.addConstrs(
        (v.E_BES_in_T[t] >= -p.k_BES_out_per_capa_ * p.E_BES_capa_ for t in T), "MAX_BES_OUT"
    )
    m.addConstr(v.E_BES_in_T[min(T)] == 0, "INI_BES")
    m.addConstr(v.E_BES_T[min(T)] == 0, "INI_BES_0")
    m.addConstr(v.E_BES_T[max(T)] == 0, "END_BES_0")


def sankey_func(sc):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    E PUR EL {r.E_pur_T.sum()}
    E EL DEM_el {p.E_dem_T.sum()}
    """


def main():
    cs = draf.CaseStudy("MO_BES", year=2019, freq="60min")
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(scen_vars=[("E_BES_capa_", "s", [100])], nParetoPoints=2)
    cs.improve_pareto_and_set_model(model_func)
    cs.optimize(logToConsole=False)
    return cs


if __name__ == "__main__":
    cs = main()
