"""An advanced version of MO-BES."""

from gurobipy import GRB, quicksum

import draf


def params_func(sc):
    T = sc.add_dim("T", infer=True)

    sc.add_par("alpha_", 0, "pareto weighting factor", "")
    sc.add_par("n_comp_", 8760 / len(T), "cost weighting factor to compensate part year analysis",
               "")
    sc.add_par("n_C_", 1, "normalization factor", "")
    sc.add_par("n_CE_", 1 / 1e4, "normalization factor", "")
    sc.add_par("AF_", 0.1, "annuitiy factor (it pays off in 1/AF_ years)", "")

    sc.add_par("P_PV_CAPx_", 0, "existing PV capacity", "kW_peak")
    sc.add_par("E_BES_CAPx_", 0, "existing BES capacity", "kW_el")

    # COSTS
    sc.add_par("c_el_peak_", 40, "peak price", "€/kW_el")
    rtp = sc.prep.add_c_GRID_RTP_T()
    tou = sc.prep.add_c_GRID_TOU_T()
    flat = sc.prep.add_c_GRID_FLAT_T()
    sc.add_par("c_GRID_T", rtp, "chosen electricity tariff", "€/kWh_el")
    sc.prep.add_c_GRID_addon_T()

    # ENVIRONMENT
    sc.prep.add_ce_GRID_T()

    # PV
    sc.prep.add_E_PV_profile_T()

    # DEMANDS
    sc.prep.add_E_dem_T(profile="G3", annual_energy=10.743e6)

    # EFFICIENCIES
    sc.add_par("eta_BES_time_", .999, "storing efficiency", "")
    sc.add_par("eta_BES_in_", .999, "loading efficiency", "")
    sc.add_par("k_BES_in_per_capa_", 1, "ratio loading power / capacity", "")

    sc.add_var("C_", "total costs", "k€/a", lb=-GRB.INFINITY)
    sc.add_var("C_op_", "operating costs", "k€/a", lb=-GRB.INFINITY)
    sc.add_var("CE_", "total emissions", "kgCO2eq/a", lb=-GRB.INFINITY)
    sc.add_var("CE_op_", "operating emissions", "kgCO2eq/a", lb=-GRB.INFINITY)
    sc.add_var("E_pur_T", "purchased electricity", "kWh_el", lb=-GRB.INFINITY)

    sc.add_var("E_PV_T", "produced el.", "kWh_el")
    sc.add_var("E_PV_OC_T", "own consumption", "kWh_el")
    sc.add_var("E_PV_FI_T", "feed-in", "kWh_el")

    sc.add_var("E_BES_T", "el. stored", "kWh_el")
    sc.add_var("E_BES_in_T", "loaded el.", "kWh_el", lb=-GRB.INFINITY)
    sc.add_var("E_BES_in_max_", "maximum loading rate el.", "kWh_el")
    sc.add_var("E_BES_out_max_", "maximum unloading rate el.", "kWh_el")

    sc.add_var("E_pur_T", "purchased el.", "kWh_el")
    sc.add_var("E_sell_T", "purchased el.", "kWh_el")
    sc.add_var("P_pur_peak_", "peak el.", "kW_el")


def model_func(m, d, p, v):
    T = d.T
    m.setObjective(((1 - p.alpha_) * v.C_op_ * p.n_C_ +
                    p.alpha_ * v.CE_op_ * p.n_CE_), GRB.MINIMIZE)

    m.addConstr(
        v.C_op_ == (v.P_pur_peak_ * p.c_el_peak_ / 1e3 + p.n_comp_ *
                    quicksum(v.E_pur_T[t] *
                             (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3
                             - v.E_sell_T[t] * p.c_GRID_T[t] / 1e3
                             for t in T)), "DEF_C_op")

    m.addConstr(v.CE_op_ == quicksum(v.E_pur_T[t] * p.ce_GRID_T[t] for t in T), "DEF_CE_op_")

    m.addConstrs((v.E_pur_T[t] + v.E_PV_OC_T[t] == p.E_dem_T[t] + v.E_BES_in_T[t] for t in T),
                 "BAL_el")

    m.addConstrs((v.E_sell_T[t] == v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_pur_T[t] <= v.P_pur_peak_ for t in T), "DEF_peakPrice")

    m.addConstrs((v.E_PV_T[t] == p.P_PV_CAPx_ * p.E_PV_profile_T[t] for t in T), "PV1")
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    m.addConstrs(
        (v.E_BES_T[t] == p.eta_BES_time_ * v.E_BES_T[t - 1] + p.eta_BES_in_ * v.E_BES_in_T[t]
         for t in T[1:]), "BAL_BES")
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_in_T[t] <= v.E_BES_in_max_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_in_T[t] >= -v.E_BES_out_max_ for t in T), "MAX_BES_OUT")
    m.addConstr(v.E_BES_in_max_ == p.E_BES_CAPx_ * p.k_BES_in_per_capa_, "DEF_E_BES_in_max_")
    m.addConstr(v.E_BES_out_max_ == p.E_BES_CAPx_ * p.k_BES_in_per_capa_, "DEF_E_BES_out_max_")
    m.addConstr(v.E_BES_T[min(T)] == 0, "INI_BES_0")
    m.addConstr(v.E_BES_T[max(T)] == 0, "END_BES_0")


def postprocess_func(r):
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T")
    r.make_pos_ent("E_pur_T")


def sankey_func(sc):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    E PUR EL {r.E_pur_T.sum()}
    E PV EL {r.E_PV_OC_T.sum()}
    E PV SELL_el {r.E_PV_FI_T.sum()}
    E EL DEM_el {p.E_dem_T.sum()}
    """


def main():
    cs = draf.CaseStudy("MO_BES2", year=2019, freq="60min")
    cs.set_datetime_filter(start="Apr-01 00", steps=24 * 10)
    sc = cs.add_REF_scen()
    sc.set_params(params_func)
    cs.add_scens([("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3)
    cs.set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs


if __name__ == "__main__":
    cs = main()
