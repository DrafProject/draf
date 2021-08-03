"""An multi-objective version of pv_bes."""

from gurobipy import GRB, Model, quicksum

import draf


def params_func(sc: draf.Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.var("C_", doc="total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_op_", doc="operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)
    sc.var("CE_op_", doc="operating emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)
    sc.prep.n_comp_()

    # Pareto
    sc.param("alpha_", data=0, doc="weighting factor", unit="")
    sc.param("n_C_", data=1, doc="normalization factor", unit="")
    sc.param("n_CE_", data=1 / 1e4, doc="normalization factor", unit="")

    # Electricity Grid
    sc.param("c_GRID_peak_", data=40, doc="peak price", unit="€/kW_el")
    sc.param("c_GRID_T", sc.prep.c_GRID_RTP_T(), doc="chosen electricity tariff", unit="€/kWh_el")
    sc.prep.c_GRID_TOU_T()
    sc.prep.c_GRID_FLAT_T()
    sc.prep.c_GRID_addon_T()
    sc.prep.ce_GRID_T()
    sc.var("E_GRID_buy_T", doc="purchased el.", unit="kWh_el")
    sc.var("E_GRID_sell_T", doc="sold el.", unit="kWh_el")
    sc.var("P_GRID_buy_peak_", doc="peak el.", unit="kW_el")

    # Demand
    sc.prep.E_dem_T(profile="G3", annual_energy=10.743e6)

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="existing PV capacity", unit="kW_peak")
    sc.prep.E_PV_profile_T()
    sc.var("E_PV_T", doc="produced el.", unit="kWh_el")
    sc.var("E_PV_OC_T", doc="own consumption", unit="kWh_el")
    sc.var("E_PV_FI_T", doc="feed-in", unit="kWh_el")

    # BES
    sc.param("E_BES_CAPx_", data=0, doc="existing BES capacity", unit="kW_el")
    sc.param("eta_BES_time_", data=0.999, doc="storing efficiency", unit="")
    sc.param("eta_BES_in_", data=0.999, doc="loading efficiency", unit="")
    sc.param("k_BES_in_per_capa_", data=1, doc="ratio loading power / capacity", unit="")
    sc.var("E_BES_T", doc="el. stored", unit="kWh_el")
    sc.var("E_BES_in_T", doc="loaded el.", unit="kWh_el", lb=-GRB.INFINITY)
    sc.var("E_BES_in_max_", doc="maximum loading rate el.", unit="kWh_el")
    sc.var("E_BES_out_max_", doc="maximum unloading rate el.", unit="kWh_el")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T

    m.setObjective(
        ((1 - p.alpha_) * v.C_op_ * p.n_C_ + p.alpha_ * v.CE_op_ * p.n_CE_), GRB.MINIMIZE
    )

    # Costs
    m.addConstr(
        v.C_op_
        == (
            v.P_GRID_buy_peak_ * p.c_GRID_peak_ / 1e3
            + p.n_comp_
            * quicksum(
                v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3
                - v.E_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3
                for t in T
            )
        ),
        "DEF_C_op",
    )

    # Carbon Emissions
    m.addConstr(v.CE_op_ == quicksum(v.E_GRID_buy_T[t] * p.ce_GRID_T[t] for t in T), "DEF_CE_op_")

    # GRID
    m.addConstrs(
        (v.E_GRID_buy_T[t] + v.E_PV_OC_T[t] == p.E_dem_T[t] + v.E_BES_in_T[t] for t in T), "BAL_el"
    )
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_buy_peak_ for t in T), "DEF_peakPrice")

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
    m.addConstrs((v.E_BES_in_T[t] <= v.E_BES_in_max_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_in_T[t] >= -v.E_BES_out_max_ for t in T), "MAX_BES_OUT")
    m.addConstr(v.E_BES_in_max_ == p.E_BES_CAPx_ * p.k_BES_in_per_capa_, "DEF_E_BES_in_max_")
    m.addConstr(v.E_BES_out_max_ == p.E_BES_CAPx_ * p.k_BES_in_per_capa_, "DEF_E_BES_out_max_")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")


def postprocess_func(r: draf.Results):
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T")
    r.make_pos_ent("E_GRID_buy_T")


def sankey_func(sc: draf.Scenario):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    E GRID EL {r.E_GRID_buy_T.sum()}
    E PV EL {r.E_PV_OC_T.sum()}
    E PV SELL_el {r.E_PV_FI_T.sum()}
    E EL DEM_el {p.E_dem_T.sum()}
    """


def main():
    cs = draf.CaseStudy("MO_BES2", year=2019, freq="60min")
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(
        scen_vars=[("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3
    )
    cs.set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
