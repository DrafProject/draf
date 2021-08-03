"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase 
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

from gurobipy import GRB, Model, quicksum

import draf


def params_func(sc: draf.Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.var("C_", doc="total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_inv_", doc="investment costs", unit="k€")
    sc.var("C_op_", doc="operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("alpha_", data=0, doc="pareto weighting factor", unit="")
    sc.param("n_C_", data=1, doc="normalization factor", unit="")
    sc.param("n_CE_", data=1, doc="normalization factor", unit="")

    # Electricity Grid
    sc.prep.ce_GRID_T()
    sc.param("c_GRID_peak_", data=40, doc="peak price", unit="€/kW_el")
    sc.var("C_GRID_peak_", doc="costs for peak price", unit="€/a")
    sc.param(
        "c_GRID_T", data=sc.prep.c_GRID_RTP_T(), doc="chosen electricity tariff", unit="€/kWh_el"
    )
    sc.prep.c_GRID_TOU_T()
    sc.prep.c_GRID_FLAT_T()
    sc.prep.c_GRID_addon_T()
    sc.var("E_GRID_buy_T", doc="bought el. from the grid", unit="kWh_el")
    sc.var("E_GRID_sell_T", doc="sold el. to the grid", unit="kWh_el")
    sc.var("P_GRID_peak_", doc="peak electricity", unit="kW_el")
    sc.var("C_GRID_buy_", doc="cost for bought electricity", unit="€/a")
    sc.var("C_GRID_sell_", doc="earnings for bought electricity", unit="€/a")

    # Demand
    sc.prep.E_dem_T(profile="G3", annual_energy=5e5)

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="existing capacity PV", unit="kW_peak")
    sc.prep.E_PV_profile_T()
    sc.var("E_PV_T", doc="produced electricity", unit="kWh_el")
    sc.var("E_PV_OC_T", doc="own consumption", unit="kWh_el")
    sc.var("E_PV_FI_T", doc="feed-in", unit="kWh_el")

    # BES
    sc.param("E_BES_CAPx_", data=0, doc="existing capacity BES", unit="kW_el")
    sc.param("eta_BES_time_", data=0.999, doc="storing efficiency", unit="")
    sc.param("eta_BES_in_", data=0.999, doc="loading efficiency", unit="")
    sc.param("k_BES_in_per_capa_", data=1, doc="ratio of charging power / capacity", unit="")
    sc.param(
        "k_BES_out_per_capa_",
        data=1,
        doc="ratio of discharging power / capacity (aka C rate)",
        unit="",
    )
    sc.var("E_BES_T", doc="electricity stored", unit="kWh_el")
    sc.var("E_BES_in_T", doc="electricity charged", unit="kWh_el", lb=-GRB.INFINITY)


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T

    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    # Costs
    m.addConstr(v.C_ == v.C_op_ + v.C_inv_, "DEF_C_")
    m.addConstr(v.C_op_ == v.C_GRID_peak_ + v.C_GRID_buy_ - v.C_GRID_sell_, "DEF_C_op_")
    m.addConstr(v.C_GRID_peak_ == v.P_GRID_peak_ * p.c_GRID_peak_ / 1e3, "DEF_c_GRID_peak_")
    m.addConstr(
        v.C_GRID_buy_
        == quicksum(v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3 for t in T),
        "DEF_C_GRID_buy_",
    )
    m.addConstr(
        v.C_GRID_sell_ == quicksum(v.E_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3 for t in T),
        "DEF_C_GRID_sell_",
    )
    m.addConstr(v.C_inv_ == 0, "DEF_C_inv_")

    # Carbon Emissions
    m.addConstr(v.CE_ == quicksum(v.E_GRID_buy_T[t] * p.ce_GRID_T[t] for t in T), "DEF_CE_op_")

    # GRID
    m.addConstrs(
        (v.E_GRID_buy_T[t] + v.E_PV_OC_T[t] == p.E_dem_T[t] + v.E_BES_in_T[t] for t in T), "BAL_el"
    )
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_peak_ for t in T), "DEF_peakPrice")

    # PV
    m.addConstrs((v.E_PV_T[t] == p.P_PV_CAPx_ * p.E_PV_profile_T[t] for t in T), "PV1")
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstrs(
        (
            v.E_BES_T[t] == p.eta_BES_time_ * v.E_BES_T[t - 1] + p.eta_BES_in_ * v.E_BES_in_T[t]
            for t in T[1:]
        ),
        "BAL_BES",
    )
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_in_T[t] <= p.E_BES_CAPx_ * p.k_BES_in_per_capa_ for t in T), "MAX_BES_IN")
    m.addConstrs(
        (v.E_BES_in_T[t] >= -(p.E_BES_CAPx_ * p.k_BES_out_per_capa_) for t in T), "MAX_BES_OUT"
    )
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")


def postprocess_func(r: draf.Results):
    """Ensures positive timeseries for sankey- and log-based plots."""
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T", "electricity uncharged")
    r.make_pos_ent("E_GRID_buy_T")


def sankey_func(sc: draf.Scenario):
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
    cs.set_time_horizon(start="Jan-02 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(
        scen_vars=[("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3
    )
    cs.improve_pareto_and_set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
