"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

from gurobipy import GRB, Model, quicksum

import draf
from draf.prep import DataBase as db


def params_func(sc: draf.Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.prep.k__comp_()
    sc.prep.k__dT_()

    # Total
    sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
    sc.param("k_PTO_C_", data=1, doc="Normalization factor")
    sc.param("k_PTO_CE_", data=1 / 1e4, doc="Normalization factor")

    # Demands
    sc.prep.P_eDem_T(profile="G3", annual_energy=5e6)
    # EL
    sc.param("c_EL_buyPeak_", data=40, doc="Peak price", unit="€/kW_el")
    sc.param("c_EL_T", data=sc.prep.c_EL_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el")
    sc.prep.c_EL_TOU_T()
    sc.prep.c_EL_FLAT_T()
    sc.prep.c_EL_addon_T()
    sc.prep.ce_EL_T()
    sc.var("P_EL_buy_T", doc="Purchased electrical power", unit="kW_el")
    sc.var("P_EL_sell_T", doc="Selling electrical power", unit="kW_el")
    sc.var("P_EL_buyPeak_", doc="Peak electrical power", unit="kW_el")

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="Existing capacity", unit="kW_peak")
    sc.prep.P_PV_profile_T(use_coords=True)
    sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")
    sc.var("P_PV_OC_T", doc="Own consumption", unit="kW_el")
    sc.var("P_PV_T", doc="Producing electrical power", unit="kW_el")

    # BES
    sc.param("E_BES_CAPx_", data=0, doc="Existing capacity", unit="kW_el")
    sc.param(from_db=db.eta_BES_cycle_)
    sc.param(from_db=db.eta_BES_time_)
    sc.param(from_db=db.k_BES_inPerCapa_)
    sc.param(from_db=db.k_BES_outPerCapa_)
    sc.var("P_BES_in_T", doc="Charging power", unit="kW_el")
    sc.var("P_BES_out_T", doc="Discharging power", unit="kW_el")
    sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T

    m.setObjective(
        ((1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_ + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_),
        GRB.MINIMIZE,
    )

    # C
    m.addConstr(v.C_TOT_ == v.C_TOT_op_, "DEF_C_")
    m.addConstr(
        v.C_TOT_op_
        == v.P_EL_buyPeak_ * p.c_EL_buyPeak_ / 1e3
        + p.k__dT_
        * (
            quicksum(v.P_EL_buy_T[t] * (p.c_EL_T[t] + p.c_EL_addon_T[t]) / 1e3 for t in T)
            - quicksum(v.P_EL_sell_T[t] * p.c_EL_T[t] / 1e3 for t in T)
        ),
        "DEF_C_TOT_op_",
    )

    # CE
    m.addConstr(
        v.CE_TOT_ == p.k__dT_ * quicksum(v.P_EL_buy_T[t] * p.ce_EL_T[t] for t in T), "DEF_CE_"
    )

    # Electricity
    m.addConstrs(
        (
            v.P_EL_buy_T[t] + v.P_PV_OC_T[t] + v.P_BES_out_T[t]
            == p.P_eDem_T[t] + v.P_BES_in_T[t] + v.P_EL_sell_T[t]
            for t in T
        ),
        "BAL_el",
    )

    # EL
    m.addConstrs((v.P_EL_sell_T[t] == v.P_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.P_EL_buy_T[t] <= v.P_EL_buyPeak_ for t in T), "DEF_peakPrice")

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs((v.P_PV_T[t] == p.P_PV_CAPx_ * p.P_PV_profile_T[t] for t in T), "PV1")
    m.addConstrs((v.P_PV_T[t] == v.P_PV_FI_T[t] + v.P_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstrs((v.P_BES_in_T[t] <= p.E_BES_CAPx_ * p.k_BES_inPerCapa_ for t in T), "MAX_BES_IN")
    m.addConstrs(
        (v.P_BES_out_T[t] <= p.E_BES_CAPx_ * p.k_BES_outPerCapa_ for t in T), "MAX_BES_OUT"
    )
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")
    m.addConstrs(
        (
            v.E_BES_T[t]
            == v.E_BES_T[t - 1] * p.eta_BES_time_
            + v.P_BES_in_T[t] * p.eta_BES_cycle_ * p.k__dT_
            - v.P_BES_out_T[t] * p.k__dT_
            for t in T[1:]
        ),
        "BAL_BES",
    )


def postprocess_func(r: draf.Results):
    r.make_pos_ent("P_EL_buy_T")


def sankey_func(sc: draf.Scenario):
    p = sc.params
    r = sc.res
    gte = sc.get_total_energy
    return f"""\
    type source target value
    E EL_buy eHub {gte(r.P_EL_buy_T)}
    E PV eHub {gte(r.P_PV_OC_T)}
    E PV EL_sell {gte(r.P_PV_FI_T)}
    E eHub BES {gte(r.P_BES_in_T)}
    E BES eDemand {gte(r.P_BES_out_T)}
    E eHub eDemand {gte(p.P_eDem_T) - gte(r.P_BES_in_T)}
    """


def main():
    cs = draf.CaseStudy("pv_bes", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(scen_vars=[("c_EL_T", "t", ["c_EL_RTP_T", "c_EL_TOU_T"])], del_REF=True)
    cs.set_model(model_func).optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
