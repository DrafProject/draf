"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

from gurobipy import GRB, Model, quicksum

import draf


def params_func(sc: draf.Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.prep.n_comp_()
    sc.var("C_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_op_", doc="Operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("alpha_", data=0, doc="Pareto weighting factor")
    sc.param("n_C_", data=1, doc="Normalization factor")
    sc.param("n_CE_", data=1 / 1e4, doc="Normalization factor")

    # Demands
    sc.prep.E_dem_T(profile="G3", annual_energy=5e6)
    # GRID
    sc.param("c_GRID_buyPeak_", data=40, doc="Peak price", unit="€/kW_el")
    sc.param(
        "c_GRID_T", data=sc.prep.c_GRID_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el"
    )
    sc.prep.c_GRID_addon_T()
    sc.prep.c_GRID_FLAT_T()
    sc.prep.c_GRID_TOU_T()
    sc.prep.ce_GRID_T()
    sc.var("E_GRID_buy_T", doc="Purchased electricity", unit="kWh_el")
    sc.var("E_GRID_sell_T", doc="Sold electricity", unit="kWh_el")
    sc.var("P_GRID_buyPeak_", doc="Peak electricity", unit="kW_el")

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="Existing capacity", unit="kW_peak")
    sc.prep.E_PV_profile_T(use_coords=True)
    sc.var("E_PV_FI_T", doc="Feed-in", unit="kWh_el")
    sc.var("E_PV_OC_T", doc="Own consumption", unit="kWh_el")
    sc.var("E_PV_T", doc="Produced electricity", unit="kWh_el")

    # BES
    sc.param("E_BES_CAPx_", data=0, doc="Existing capacity", unit="kW_el")
    sc.param("eta_BES_in_", data=0.999, doc="Discharging efficiency")
    sc.param("eta_BES_time_", data=0.999, doc="Charging efficiency")
    sc.param("k_BES_inPerCapa_", data=1, doc="Ratio charging power / capacity")
    sc.param("k_BES_outPerCapa_", data=1, doc="Ratio discharging power / capacity")
    sc.var("E_BES_in_T", doc="Charged electricity", unit="kWh_el")
    sc.var("E_BES_inMax_", doc="Maximum charging rate", unit="kWh_el")
    sc.var("E_BES_out_T", doc="Discharged electricity", unit="kWh_el")
    sc.var("E_BES_outMax_", doc="Maximum discharging rate", unit="kWh_el")
    sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T

    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    # C
    m.addConstr(v.C_ == v.C_op_, "DEF_C_")
    m.addConstr(
        v.C_op_
        == v.P_GRID_buyPeak_ * p.c_GRID_buyPeak_ / 1e3
        + quicksum(v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3 for t in T)
        - quicksum(v.E_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3 for t in T),
        "DEF_C_op_",
    )

    # CE
    m.addConstr(v.CE_ == quicksum(v.E_GRID_buy_T[t] * p.ce_GRID_T[t] for t in T), "DEF_CE_")

    # Electricity
    m.addConstrs(
        (
            v.E_GRID_buy_T[t] + v.E_PV_OC_T[t] + v.E_BES_out_T[t] == p.E_dem_T[t] + v.E_BES_in_T[t]
            for t in T
        ),
        "BAL_el",
    )

    # GRID
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_buyPeak_ for t in T), "DEF_peakPrice")

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs((v.E_PV_T[t] == p.P_PV_CAPx_ * p.E_PV_profile_T[t] for t in T), "PV1")
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstr(v.E_BES_inMax_ == p.E_BES_CAPx_ * p.k_BES_inPerCapa_, "DEF_E_BES_inMax_")
    m.addConstr(v.E_BES_outMax_ == p.E_BES_CAPx_ * p.k_BES_outPerCapa_, "DEF_E_BES_outMax_")
    m.addConstrs((v.E_BES_in_T[t] <= v.E_BES_inMax_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_out_T[t] <= v.E_BES_outMax_ for t in T), "MAX_BES_OUT")
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")
    m.addConstrs(
        (
            v.E_BES_T[t]
            == v.E_BES_T[t - 1] * p.eta_BES_time_
            + v.E_BES_in_T[t] * p.eta_BES_in_
            - v.E_BES_out_T[t]
            for t in T[1:]
        ),
        "BAL_BES",
    )


def postprocess_func(r: draf.Results):
    r.make_pos_ent("E_GRID_buy_T")


def sankey_func(sc: draf.Scenario):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    E GRID_buy eHub {r.E_GRID_buy_T.sum()}
    E PV eHub {r.E_PV_OC_T.sum()}
    E PV GRID_sell {r.E_PV_FI_T.sum()}
    E eHub BES {r.E_BES_in_T.sum()}
    E BES eDemand {r.E_BES_out_T.sum()}
    E eHub eDemand {p.E_dem_T.sum()- r.E_BES_in_T.sum()}
    """


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(
        scen_vars=[("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3
    )
    cs.set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
