"""A model with a Photovoltaic and a Battery Energy Storage System. It is used in the showcase
see: https://mfleschutz.github.io/draf-showcase/#/2
"""

from gurobipy import GRB, Model, quicksum

import draf
from draf import Dimensions, Params, Results, Scenario, Vars
from draf.prep import DataBase as db


def params_func(sc: Scenario):

    # Dimensions
    sc.dim("T", infer=True)

    # General
    sc.prep.k__PartYearComp_()
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

    # EG
    sc.param("c_EG_buyPeak_", data=50, doc="Peak price", unit="€/kW_el")
    sc.param("c_EG_T", data=sc.prep.c_EG_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el")
    sc.prep.c_EG_TOU_T()
    sc.prep.c_EG_FLAT_T()
    sc.prep.c_EG_addon_T()
    sc.prep.ce_EG_T()
    sc.var("P_EG_buy_T", doc="Purchased electrical power", unit="kW_el")
    sc.var("P_EG_sell_T", doc="Selling electrical power", unit="kW_el")
    sc.var("P_EG_buyPeak_", doc="Peak electrical power", unit="kW_el")
    sc.param(
        "c_OC_",
        data=0.4 * 0.0688,
        doc="Renewable Energy Law (EEG) levy on own consumption",
        unit="€/kWh_el",
    )

    # PV
    sc.param("P_PV_CAPx_", data=100, doc="Existing capacity", unit="kW_peak")

    sc.prep.P_PV_profile_T(use_coords=True)
    sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")
    sc.var("P_PV_OC_T", doc="Own consumption", unit="kW_el")
    sc.var("P_PV_T", doc="Producing electrical power", unit="kW_el")
    # BES
    sc.param("E_BES_CAPx_", data=100, doc="Existing capacity", unit="kWh_el")
    sc.param(from_db=db.eta_BES_cycle_)
    sc.param(from_db=db.eta_BES_time_)
    sc.param(from_db=db.k_BES_inPerCap_)
    sc.param(from_db=db.k_BES_outPerCap_)
    sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")
    sc.var("P_BES_in_T", doc="Charging power", unit="kW_el")
    sc.var("P_BES_out_T", doc="Discharging power", unit="kW_el")


def model_func(sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):

    m.setObjective(
        ((1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_ + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_),
        GRB.MINIMIZE,
    )

    # C
    m.addConstr(v.C_TOT_ == v.C_TOT_op_, "DEF_C_")
    m.addConstr(
        v.C_TOT_op_
        == v.P_EG_buyPeak_ * p.c_EG_buyPeak_ / 1e3
        + p.k__PartYearComp_
        * p.k__dT_
        * (
            quicksum(v.P_EG_buy_T[t] * (p.c_EG_T[t] + p.c_EG_addon_T[t]) for t in d.T)
            - quicksum(v.P_EG_sell_T[t] * p.c_EG_T[t] for t in d.T)
            + v.P_PV_OC_T.sum() * p.c_OC_
        )
        / 1e3,
        "DEF_C_TOT_op_",
    )

    # CE
    m.addConstr(
        v.CE_TOT_
        == p.k__PartYearComp_
        * p.k__dT_
        * quicksum((v.P_EG_buy_T[t] - v.P_EG_sell_T[t]) * p.ce_EG_T[t] for t in d.T),
        "DEF_CE_",
    )

    # Electricity
    m.addConstrs(
        (
            v.P_EG_buy_T[t] + v.P_PV_OC_T[t] + v.P_BES_out_T[t]
            == p.P_eDem_T[t] + v.P_BES_in_T[t] + v.P_EG_sell_T[t]
            for t in d.T
        ),
        "BAL_el",
    )

    # EG
    m.addConstrs((v.P_EG_sell_T[t] == v.P_PV_FI_T[t] for t in d.T), "DEF_E_sell")
    m.addConstrs((v.P_EG_buy_T[t] <= v.P_EG_buyPeak_ for t in d.T), "DEF_peakPrice")

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs((v.P_PV_T[t] == p.P_PV_CAPx_ * p.P_PV_profile_T[t] for t in d.T), "PV1")
    m.addConstrs((v.P_PV_T[t] == v.P_PV_FI_T[t] + v.P_PV_OC_T[t] for t in d.T), "PV_OC_FI")

    # BES
    m.addConstrs((v.P_BES_in_T[t] <= p.E_BES_CAPx_ * p.k_BES_inPerCap_ for t in d.T), "MAX_BES_IN")
    m.addConstrs(
        (v.P_BES_out_T[t] <= p.E_BES_CAPx_ * p.k_BES_outPerCap_ for t in d.T), "MAX_BES_OUT"
    )
    m.addConstrs((v.E_BES_T[t] <= p.E_BES_CAPx_ for t in d.T), "MAX_BES_E")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(d.T), max(d.T)]), "INI_BES")
    m.addConstrs(
        (
            v.E_BES_T[t]
            == v.E_BES_T[t - 1] * p.eta_BES_time_
            + v.P_BES_in_T[t] * p.eta_BES_cycle_ * p.k__dT_
            - v.P_BES_out_T[t] * p.k__dT_
            for t in d.T[1:]
        ),
        "BAL_BES",
    )


def postprocess_func(r: Results):
    r.make_pos_ent("P_EG_buy_T")


def sankey_func(sc: Scenario):
    p = sc.params
    r = sc.res
    gte = sc.get_total_energy
    return f"""\
    type source target value
    E EG_buy eHub {gte(r.P_EG_buy_T)}
    E PV eHub {gte(r.P_PV_OC_T)}
    E PV EG_sell {gte(r.P_PV_FI_T)}
    E eHub BES {gte(r.P_BES_in_T)}
    E BES eDemand {gte(r.P_BES_out_T)}
    E eHub eDemand {gte(p.P_eDem_T) - gte(r.P_BES_in_T)}
    """


def main():
    cs = draf.CaseStudy("pv_bes", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Aug-01 00:00", steps=24)
    cs.add_REF_scen().set_params(params_func).update_params(E_BES_CAPx_=100, P_PV_CAPx_=1e3)
    cs.add_scens([("c_EG_T", "t", ["c_EG_TOU_T", "c_EG_FLAT_T"])])
    cs.set_model(model_func).optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
