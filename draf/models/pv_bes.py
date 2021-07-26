"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase 
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

from gurobipy import GRB, quicksum

import draf


def params_func(sc):
    """Defines model parameters and variables with bounds, type, units, descriptions, etc."""

    # DIMENSIONS
    T = sc.add_dim("T", infer=True)

    # COSTS
    sc.add_var("C_", "total costs", "k€/a", lb=-GRB.INFINITY)
    sc.add_var("C_inv_", "investment costs", "k€")
    sc.add_var("C_op_", "operating costs", "k€/a", lb=-GRB.INFINITY)

    # EMISSIONS
    sc.add_var("CE_", "total emissions", "kgCO2eq/a", lb=-GRB.INFINITY)

    # GENERAL
    sc.add_par(name="alpha_", data=0, doc="pareto weighting factor", unit="")
    sc.add_par("n_C_", 1, "normalization factor", "")
    sc.add_par("n_CE_", 1, "normalization factor", "")

    # ELECTRICITY GRID
    sc.prep.add_ce_GRID_T()
    sc.add_par("c_GRID_peak_", 40, "peak price", "€/kW_el")
    sc.add_var("C_GRID_peak_", "costs for peak price", "€/a")
    rtp = sc.prep.add_c_GRID_RTP_T()
    tou = sc.prep.add_c_GRID_TOU_T()
    # flat = sc.prep.add_c_GRID_FLAT_T()
    sc.add_par("c_GRID_T", rtp, "chosen electricity tariff", "€/kWh_el")
    sc.prep.add_c_GRID_addon_T()
    sc.add_var("E_GRID_buy_T", "bought el. from the grid", "kWh_el")
    sc.add_var("E_GRID_sell_T", "sold el. to the grid", "kWh_el")
    sc.add_var("P_GRID_peak_", "peak electricity", "kW_el")
    sc.add_var("C_GRID_buy_", "cost for bought electricity", "€/a")
    sc.add_var("C_GRID_sell_", "earnings for bought electricity", "€/a")

    # INVEST
    # sc.add_par("AF_", 0.1, "annuitiy factor (it pays off in 1/AF_ years)", "")

    # DEMANDS
    sc.prep.add_E_dem_T(profile="G3", annual_energy=5e5)

    # PV
    sc.add_par("P_PV_CAPx_", 0, "existing capacity PV", "kW_peak")
    sc.prep.add_E_PV_profile_T()
    sc.add_var("E_PV_T", "produced electricity", "kWh_el")
    sc.add_var("E_PV_OC_T", "own consumption", "kWh_el")
    sc.add_var("E_PV_FI_T", "feed-in", "kWh_el")

    # BES
    sc.add_par("E_BES_CAPx_", 0, "existing capacity BES", "kW_el")
    sc.add_par("eta_BES_time_", 0.999, "storing efficiency", "")
    sc.add_par("eta_BES_in_", 0.999, "loading efficiency", "")
    sc.add_par("k_BES_in_per_capa_", 1, "ratio of charging power / capacity", "")
    sc.add_par("k_BES_out_per_capa_", 1, "ratio of discharging power / capacity (aka C rate)", "")
    sc.add_var("E_BES_T", "electricity stored", "kWh_el")
    sc.add_var("E_BES_in_T", "electricity charged", "kWh_el", lb=-GRB.INFINITY)


def model_func(m, d, p, v):
    """Sets model constraints. Arguments: (m)odel, (d)imensions, (p)arameters, and (v)ariables."""

    # ALIAS
    T = d.T

    # OBJECTIVE
    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    # COST BALANCE
    m.addConstr(v.C_ == v.C_op_ + v.C_inv_, "DEF_C_")
    m.addConstr(v.C_op_ == v.C_GRID_peak_ + v.C_GRID_buy_ - v.C_GRID_sell_, "DEF_C_op_")
    m.addConstr(v.C_GRID_peak_ == v.P_GRID_peak_ * p.c_GRID_peak_ / 1e3, "DEF_C_el_peak_")
    m.addConstr(
        v.C_GRID_buy_ == quicksum(v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3 for t in T),
        "DEF_C_GRID_buy_",
    )
    m.addConstr(
        v.C_GRID_sell_ == quicksum(v.E_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3 for t in T), "DEF_C_GRID_sell_",
    )
    m.addConstr(v.C_inv_ == 0, "DEF_C_inv_")

    # CARBON BALANCE
    m.addConstr(v.CE_ == quicksum(v.E_GRID_buy_T[t] * p.ce_GRID_T[t] for t in T), "DEF_CE_op_")

    # ELECTRICITY BALANCE
    m.addConstrs((v.E_GRID_buy_T[t] + v.E_PV_OC_T[t] == p.E_dem_T[t] + v.E_BES_in_T[t] for t in T), "BAL_el")
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_peak_ for t in T), "DEF_peakPrice")

    # PV
    m.addConstrs((v.E_PV_T[t] == p.P_PV_CAPx_ * p.E_PV_profile_T[t] for t in T), "PV1")
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstrs(
        (v.E_BES_T[t] == p.eta_BES_time_ * v.E_BES_T[t - 1] + p.eta_BES_in_ * v.E_BES_in_T[t] for t in T[1:]),
        "BAL_BES",
    )
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_in_T[t] <= p.E_BES_CAPx_ * p.k_BES_in_per_capa_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_in_T[t] >= -(p.E_BES_CAPx_ * p.k_BES_out_per_capa_) for t in T), "MAX_BES_OUT")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")


def postprocess_func(r):
    """Ensures positive timeseries for sankey- and log-based plots."""
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T", "electricity uncharged")
    r.make_pos_ent("E_GRID_buy_T")


def sankey_func(sc):
    """Specifies the energy flows for use in sankey plots"""
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    E GRID_buy ElectricityHub {r.E_GRID_buy_T.sum()}
    E PV ElectricityHub {r.E_PV_OC_T.sum()}
    E PV GRID_sell {r.E_PV_FI_T.sum()}
    E ElectricityHub BES {r.E_BES_in_T.sum()}
    E BES ElectricityDemand {r.E_BES_out_T.sum()}
    E ElectricityHub ElectricityDemand {p.E_dem_T.sum()- r.E_BES_in_T.sum()}
    """


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min")
    cs.set_datetime_filter(start="Jan-02 00", steps=24 * 2)
    sc = cs.add_REF_scen()
    sc.set_params(params_func)
    cs.add_scens([("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3)
    cs.improve_pareto_and_set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs


if __name__ == "__main__":
    cs = main()
