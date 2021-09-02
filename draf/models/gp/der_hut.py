"""DER-HUT: Distributed Energy Resources - Heat Upgrading Technologies"""

import pandas as pd
from gurobipy import GRB, Model, quicksum

import draf
from draf.model_builder import collectors
from draf.prep import DataBase as db


def params_func(sc: draf.Scenario):

    # Alias
    p = sc.params
    d = sc.dims

    # Dimensions
    sc.dim("T", infer=True)
    sc.dim("F", ["ng", "bio"], doc="Types of fuel")
    sc.dim("C", ["30", "65"], doc="Condensing temperature levels")
    sc.dim("H", ["60/40", "90/70"], doc="Heating temperature levels (inlet / outlet) in °C")
    sc.dim("N", ["7/12", "30/35"], doc="Cooling temperature levels (inlet / outlet) in °C")

    # General
    sc.param("k__AF_", 0.1, doc="Annuity factor (it pays off in 1/k__AF_ years)")
    sc.prep.k__comp_()
    sc.prep.k__dT_()

    # Total
    sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_TOT_inv_", doc="Total investment costs", unit="k€")
    sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_TOT_maint_", doc="Total annual maintainance cost", unit="k€")
    sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
    sc.param("k_PTO_C_", data=1, doc="Pareto normalization factor")
    sc.param("k_PTO_CE_", data=1 / 1e4, doc="Pareto normalization factor")

    # Fuels
    sc.param(from_db=db.c_FUEL_F)
    sc.param(from_db=db.ce_FUEL_F)

    # Demands
    sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)
    sc.prep.dQ_cDem_TN()
    p.dQ_cDem_TN.loc[:, "7/12"] = sc.prep.dQ_dem_C_T(annual_energy=1.7e3).values
    sc.prep.dQ_hDem_TH()
    p.dQ_hDem_TH.loc[:, "60/40"] = sc.prep.dQ_dem_H_T(annual_energy=1.7e6).values
    sc.param(
        "T_hDem_out_H", data=273 + pd.Series([40, 70], d.H), doc="Heating outlet temp.", unit="K"
    )
    sc.param(
        "T_hDem_in_H", data=273 + pd.Series([60, 90], d.H), doc="Heating inlet temp.", unit="K"
    )
    sc.param("T_cDem_in_N", data=273 + pd.Series([7, 30], d.N), doc="Cooling inlet temp.", unit="K")
    sc.param(
        "T_cDem_out_N", data=273 + pd.Series([12, 35], d.N), doc="Cooling outlet temp.", unit="K"
    )

    # GRID
    sc.param("c_GRID_peak_", data=50, doc="Peak price", unit="€/kW_el")
    sc.param(
        "c_GRID_T", data=sc.prep.c_GRID_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el"
    )
    sc.prep.c_GRID_addon_T()
    sc.prep.ce_GRID_T()
    sc.var("P_GRID_buy_T", doc="Purchasing power", unit="kW_el")
    sc.var("P_GRID_sell_T", doc="Sell power", unit="kW_el")
    sc.var("P_GRID_buyPeak_", doc="Peak power", unit="kW_el")

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="Existing capacity", unit="kW_peak")
    sc.param("z_PV_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_PV_inv_())
    sc.param(from_db=db.k_PV_maint_)
    sc.prep.P_PV_profile_T(use_coords=True)
    sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")
    sc.var("P_PV_OC_T", doc="Own consumption", unit="kW_el")
    sc.var("P_PV_T", doc="Producing power", unit="kW_el")
    sc.var("P_PV_CAPn_", doc="New capacity", unit="kW_el", ub=1e20 * p.z_PV_)

    # BES
    sc.param("k_BES_inPerCapa_", data=1.0, doc="Ratio charging power / capacity")
    sc.param("k_BES_outPerCapa_", data=1.0, doc="Ratio discharging power / capacity")
    sc.param("z_BES_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.eta_BES_cycle_)
    sc.param(from_db=db.eta_BES_time_)
    sc.param(from_db=db.funcs.c_BES_inv_(estimated_size=100, which="mean"))
    sc.param(from_db=db.k_BES_maint_)
    sc.var("E_BES_CAPn_", doc="New capacity", unit="kWh_el")
    sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")
    sc.var("P_BES_in_T", doc="Charging power", unit="kW_el")
    sc.var("P_BES_inMax_", doc="Maximum charging power", unit="kW_el")
    sc.var("P_BES_out_T", doc="Discharging power", unit="kW_el")
    sc.var("P_BES_outMax_", doc="Maximum discharging power", unit="kW_el")

    # HP
    sc.param("T__amb_", data=273 + 25)
    sc.param(
        "T_HP_C_C",
        data=pd.Series([p.T__amb_ + 5, 273 + 65], d.C),
        doc="Condensation side temp.",
        unit="K",
    )
    sc.param("T_HP_E_N", data=p.T_cDem_in_N - 5, doc="Evaporation side temp.", unit="K")
    sc.param("eta_HP_", data=0.5, doc="Ratio of reaching the ideal COP (exergy efficiency")
    sc.param(
        "cop_HP_carnotH_CN",
        data=pd.Series(
            {(c, n): p.T_HP_C_C[c] / (p.T_HP_C_C[c] - p.T_HP_E_N[n]) for c in d.C for n in d.N}
        ),
        doc="Ideal Carnot heating coefficient of performance (COP)",
    )
    sc.param(
        "cop_HP_CN",
        data=pd.Series(
            {
                (c, n): 100
                if p.T_HP_C_C[c] <= p.T_HP_E_N[n]
                else p.cop_HP_carnotH_CN[c, n] * p.eta_HP_
                for c in d.C
                for n in d.N
            }
        ),
        doc="Real heating COP",
    )
    sc.param("dQ_HP_CAPx_", data=5000, doc="Existing heating capacity", unit="kW_th")
    sc.param("dQ_HP_max_", data=1e6, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_th")
    sc.param("z_HP_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_HP_inv_())
    sc.param(from_db=db.k_HP_maint_)
    sc.var("P_HP_TCN", doc="Consuming power", unit="kW_el")
    sc.var("dQ_HP_CAPn_", doc="New heating capacity", unit="kW_th", ub=1e6 * p.z_HP_)
    sc.var("dQ_HP_Cond_TCN", doc="Heat flow released on condensation side", unit="kW_th")
    sc.var("dQ_HP_Eva_TCN", doc="Heat flow absorbed on evaporation side", unit="kW_th")
    sc.var("Y_HP_TCN", doc="1, If source and sink are connected at time-step", vtype=GRB.BINARY)

    # HOB
    sc.param("eta_HOB_", data=0.9, doc="Thermal efficiency", unit="kWh_th/kWh")
    sc.param(from_db=db.funcs.c_HOB_inv_())
    sc.param("dQ_HOB_CAPx_", data=10000, doc="Existing capacity", unit="kW_th")
    sc.param(from_db=db.k_HOB_maint_)
    sc.var("F_HOB_TF", doc="Input fuel flow", unit="kW")
    sc.var("dQ_HOB_CAPn_", doc="New capacity", unit="kW_th")
    sc.var("dQ_HOB_T", doc="Ouput heat flow", unit="kW_th")

    # H2H1
    sc.var("dQ_H2H1_T", doc="Heat down-grading", unit="kWh_th")

    # P2H
    sc.param("Q_P2H_CAPx_", data=10000, doc="Existing capacity", unit="kW_th")
    sc.param(from_db=db.eta_P2H_)
    sc.var("P_P2H_T", doc="Consuming power", unit="kW_el")
    sc.var("dQ_P2H_T", doc="Producing heat flow", unit="kW_th")

    # CHP
    sc.param(from_db=db.funcs.eta_CHP_th_(fuel="ng"))
    sc.param(from_db=db.funcs.eta_CHP_el_(fuel="ng"))
    sc.param("P_CHP_CAPx_", data=0, doc="Existing capacity", unit="kW_el")
    sc.param("z_CHP_", data=0, doc="If new capacity is allowed")
    sc.param("z_CHP_minPL_", data=1, doc="If minimal part load is modeled.")
    sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
    sc.param(from_db=db.k_CHP_maint_)
    sc.var("P_CHP_FI_T", doc="Feed-in", unit="kW_el")
    sc.var("P_CHP_OC_T", doc="Own consumption", unit="kW_el")
    sc.var("P_CHP_T", doc="Producing power", unit="kW_el")
    sc.var("F_CHP_TF", doc="Consumed fuel flow", unit="kW")
    sc.var("P_CHP_CAPn_", doc="New capacity", unit="kW_el", ub=1e6 * p.z_CHP_)
    sc.var("dQ_CHP_T", doc="Producing heat flow", unit="kW_th")
    sc.param("P_CHP_max_", data=1e6, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_el")
    if p.z_CHP_minPL_:
        sc.param("k_CHP_minPL_", data=0.5, doc="Minimal allowed part load")
        sc.var("Y_CHP_T", doc="If in operation", vtype=GRB.BINARY)

    # TES
    sc.param("eta_TES_time_", data=0.995, doc="Storing efficiency")
    sc.param("k_TES_inPerCapa_", data=0.5, doc="Ratio loading power / capacity")
    sc.param("k_TES_outPerCapa_", data=0.5, doc="Ratio loading power / capacity")
    sc.param("z_TES_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_TES_inv_(estimated_size=100, temp_spread=40))
    sc.param(from_db=db.k_TES_maint_)
    sc.var("Q_TES_CAPn_H", doc="New capacity", unit="kWh_th", ub=1e7 * p.z_TES_)
    sc.var("dQ_TES_in_TH", doc="Storage input heat flow", unit="kW_th", lb=-GRB.INFINITY)
    sc.var("Q_TES_TH", doc="Stored heat", unit="kWh_th")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T
    F = d.F
    N = d.N
    C = d.C
    H = d.H

    m.setObjective(
        ((1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_ + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_),
        GRB.MINIMIZE,
    )

    # C
    # Note: all energy-specific costs are divided by 1e3 to get k€ and therefore to
    # ensure smaller coefficient ranges which speeds up the optimization.
    m.addConstr(v.C_TOT_ == v.C_TOT_op_ + p.k__AF_ * v.C_TOT_inv_, "DEF_C_")
    m.addConstr(
        v.C_TOT_op_
        == (
            v.P_GRID_buyPeak_ * p.c_GRID_peak_ / 1e3
            + v.C_TOT_maint_
            + p.k__comp_
            * p.k__dT_
            * quicksum(
                v.P_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3
                - v.P_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3
                + quicksum((v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) * p.c_FUEL_F[f] / 1e3 for f in F)
                for t in T
            )
        ),
        "DEF_C_TOT_op_",
    )
    m.addConstr((v.C_TOT_inv_ == collectors.C_TOT_inv_(p, v) / 1e3))
    m.addConstr((v.C_TOT_maint_ == collectors.C_TOT_maint_(p, v) / 1e3), "DEF_C_TOT_maint_")

    # CE
    m.addConstr(
        v.CE_TOT_
        == p.k__comp_
        * p.k__dT_
        * quicksum(
            p.ce_GRID_T[t] * (v.P_GRID_buy_T[t] - v.P_GRID_sell_T[t])
            + quicksum(p.ce_FUEL_F[f] * (v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) for f in F)
            for t in T
        )
        / 1e6,
        "DEF_CE_",
    )

    # Electricity
    m.addConstrs(
        (
            v.P_GRID_buy_T[t] + v.P_CHP_OC_T[t] + v.P_PV_OC_T[t] + v.P_BES_out_T[t]
            == p.P_eDem_T[t]
            + v.P_HP_TCN.sum(t, "*", "*")
            + v.P_P2H_T[t]
            + v.P_BES_in_T[t]
            + v.P_GRID_sell_T[t]
            for t in T
        ),
        "BAL_el",
    )

    # Heat
    m.addConstrs(
        (
            v.dQ_HOB_T[t] + v.dQ_CHP_T[t]
            == p.dQ_hDem_TH[t, "90/70"] + v.dQ_H2H1_T[t] + v.dQ_TES_in_TH[t, "90/70"]
            for t in T
        ),
        "BAL_H2",
    )

    # COOL
    m.addConstrs(
        (p.dQ_cDem_TN[t, n] == v.dQ_HP_Eva_TCN.sum(t, "*", n) for t in T for n in N), "BAL_HP_N1"
    )

    # GRID
    m.addConstrs((v.P_GRID_sell_T[t] == v.P_CHP_FI_T[t] + v.P_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.P_GRID_buy_T[t] <= v.P_GRID_buyPeak_ for t in T), "DEF_peakPrice")

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs(
        (v.P_PV_T[t] == (p.P_PV_CAPx_ + v.P_PV_CAPn_) * p.P_PV_profile_T[t] for t in T), "PV1"
    )
    m.addConstrs((v.P_PV_T[t] == v.P_PV_FI_T[t] + v.P_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstr(v.P_BES_inMax_ == v.E_BES_CAPn_ * p.k_BES_inPerCapa_, "DEF_P_BES_inMax_")
    m.addConstr(v.P_BES_outMax_ == v.E_BES_CAPn_ * p.k_BES_outPerCapa_, "DEF_P_BES_outMax_")
    m.addConstrs((v.P_BES_in_T[t] <= v.P_BES_inMax_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.P_BES_out_T[t] <= v.P_BES_outMax_ for t in T), "MAX_BES_OUT")
    m.addConstrs((v.E_BES_T[t] <= v.E_BES_CAPn_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_T[t] == 0 for t in [min(T), max(T)]), "INI_BES")
    m.addConstrs(
        (
            v.E_BES_T[t]
            == v.E_BES_T[t - 1] * p.eta_BES_time_
            + v.P_BES_in_T[t] * p.eta_BES_cycle_
            - v.P_BES_out_T[t]
            for t in T[1:]
        ),
        "BAL_BES",
    )

    # CHP
    m.addConstrs(
        (v.P_CHP_T[t] == p.eta_CHP_el_ * quicksum(v.F_CHP_TF[t, f] for f in F) for t in T), "CHP_E"
    )
    m.addConstrs(
        (v.dQ_CHP_T[t] == p.eta_CHP_th_ * quicksum(v.F_CHP_TF[t, f] for f in F) for t in T), "CHP_Q"
    )
    m.addConstrs((v.P_CHP_T[t] <= p.P_CHP_CAPx_ + v.P_CHP_CAPn_ for t in T), "CHP_CAP")
    m.addConstrs((v.P_CHP_T[t] == v.P_CHP_FI_T[t] + v.P_CHP_OC_T[t] for t in T), "CHP_OC_FI")
    if p.z_CHP_minPL_:
        m.addConstrs((v.P_CHP_T[t] <= v.Y_CHP_T[t] * p.P_CHP_max_ for t in T), "DEF_Y_CHP_T")
        m.addConstrs(
            (
                v.P_CHP_T[t]
                >= p.k_CHP_minPL_ * (p.P_CHP_CAPx_ + v.P_CHP_CAPn_)
                - p.P_CHP_max_ * (1 - v.Y_CHP_T[t])
                for t in T
            ),
            "DEF_CHP_minPL",
        )

    # HOB
    m.addConstrs((v.dQ_HOB_T[t] == v.F_HOB_TF.sum(t, "*") * p.eta_HOB_ for t in T), "BAL_HOB")
    m.addConstrs((v.dQ_HOB_T[t] <= p.dQ_HOB_CAPx_ + v.dQ_HOB_CAPn_ for t in T), "CAP_HOB")

    # P2H
    m.addConstrs((v.dQ_P2H_T[t] == p.eta_P2H_ * v.P_P2H_T[t] for t in T), "BAL_P2H")
    m.addConstrs((v.dQ_P2H_T[t] <= p.Q_P2H_CAPx_ for t in T), "CAP_P2H")

    # HP
    m.addConstrs(
        (
            v.dQ_HP_Cond_TCN[t, c, n] == v.P_HP_TCN[t, c, n] * p.cop_HP_CN[c, n]
            for t in T
            for c in C
            for n in N
        ),
        "HP_BAL_1",
    )
    m.addConstrs(
        (
            v.dQ_HP_Cond_TCN[t, c, n] == v.dQ_HP_Eva_TCN[t, c, n] + v.P_HP_TCN[t, c, n]
            for t in T
            for c in C
            for n in N
        ),
        "HP_BAL_2",
    )
    m.addConstrs(
        (
            v.dQ_HP_Cond_TCN[t, c, n] <= v.Y_HP_TCN[t, c, n] * p.dQ_HP_max_
            for t in T
            for c in C
            for n in N
        ),
        "HP_BIGM",
    )
    m.addConstrs(
        (
            v.dQ_HP_Cond_TCN[t, c, n] <= v.dQ_HP_CAPn_ + p.dQ_HP_CAPx_
            for t in T
            for c in C
            for n in N
        ),
        "HP_CAP",
    )
    m.addConstrs((v.Y_HP_TCN.sum(t, "*", "*") <= 1 for t in T), "HP_maxOneOperatingMode")

    # TES
    m.addConstrs(
        (
            v.Q_TES_TH[t, h]
            == v.Q_TES_TH[t - 1, h] * p.eta_TES_time_ * p.k__dT_ + v.dQ_TES_in_TH[t - 1, h]
            for t in T[1:]
            for h in H
        ),
        "TES1",
    )
    m.addConstrs((v.Q_TES_TH[t, h] <= v.Q_TES_CAPn_H[h] for t in T for h in H), "TES3")
    m.addConstrs(
        (v.dQ_TES_in_TH[t, h] <= v.Q_TES_CAPn_H[h] * p.k_TES_inPerCapa_ for t in T for h in H),
        "MAX_TES_IN",
    )
    m.addConstrs(
        (v.dQ_TES_in_TH[t, h] >= -(v.Q_TES_CAPn_H[h] * p.k_TES_outPerCapa_) for t in T for h in H),
        "MAX_TES_OUT",
    )
    m.addConstrs((v.Q_TES_TH[t, h] == 0 for t in [min(T), max(T)] for h in H), "INI_TES")


def postprocess_func(r: draf.Results):
    r.make_pos_ent("P_GRID_buy_T")
    r.make_pos_ent("P_CHP_OC_T")
    r.make_pos_ent("dQ_TES_in_TH", "dQ_TES_out_TH")


def sankey_func(sc: draf.Scenario):
    p = sc.params
    r = sc.res
    gte = sc.get_total_energy
    return f"""\
    type source target value
    F GAS CHP {gte(r.F_CHP_TF)}
    F GAS HOB {gte(r.F_HOB_TF)}
    E GRID EL {gte(r.P_GRID_buy_T)}
    E PV EL {gte(r.P_PV_OC_T)}
    E CHP EL {gte(r.P_CHP_OC_T)}
    E PV SELL_el {gte(r.P_PV_FI_T)}
    E CHP SELL_el {gte(r.P_CHP_FI_T)}
    E EL HP {gte(r.P_HP_TCN)}
    E EL DEM_el {gte(p.P_eDem_T)}
    Q CHP H2 {gte(r.dQ_CHP_T)}
    Q HOB H2 {gte(r.dQ_HOB_T)}
    Q HP DEM_H1 {gte(r.dQ_HP_Cond_TCN)[:,3,:]}
    Q H2 H1 {gte(r.dQ_H2H1_T)}
    Q H1 DEM_H1 {gte(r.dQ_H2H1_T)}
    Q H2 DEM_H2 {gte(p.dQ_hDem_TH)[:,2]}
    Q H1 SELL_th {gte(r.dQ_sell_TH)[:,1]}
    Q H2 SELL_th {gte(r.dQ_sell_TH)[:,2]}
    Q H1 LOSS_th {gte(r.dQ_TES_in_TH[:,1]) - gte(r.dQ_TES_out_TH[:,1])}
    Q H2 LOSS_th {gte(r.dQ_TES_in_TH[:,2]) - gte(r.dQ_TES_out_TH[:,2])}
    Q HOB LOSS_th {gte(r.F_HOB_TF) - gte(r.dQ_HOB_T)}
    Q DEM_N1 HP {gte(r.dQ_HP_Eva_TCN[:,:,1])}
    Q DEM_N2 HP {gte(r.dQ_HP_Eva_TCN[:,:,2])}
    """


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
