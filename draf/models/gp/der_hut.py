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
    sc.dim("C", ["25", "35", "60"], doc="Condensing temperature levels")
    sc.dim("H", ["60/40", "90/70"], doc="Heating temperature levels")
    sc.dim("N", ["7/12", "30/35"], doc="Cooling temperature levels")

    # General
    sc.param("AF_", 0.1, doc="Annuity factor (it pays off in 1/AF_ years)")
    sc.prep.n_comp_()
    sc.var("C_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_inv_", doc="Investment costs", unit="k€")
    sc.var("C_op_", doc="Operating costs", unit="k€/a", lb=-GRB.INFINITY)
    sc.var("C_RMI_", doc="Annual maintainance cost", unit="k€")
    sc.var("CE_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("alpha_", data=0, doc="Pareto weighting factor")
    sc.param("n_C_", data=1, doc="Normalization factor")
    sc.param("n_CE_", data=1 / 1e4, doc="Normalization factor")

    # Fuels
    sc.param(from_db=db.c_FUEL_F)
    sc.param(from_db=db.ce_FUEL_F)

    # Demands
    sc.prep.E_dem_T(profile="G1", annual_energy=5e6)
    sc.prep.Q_dem_C_TN()
    p.Q_dem_C_TN.loc[:, "7/12"] = sc.prep.Q_dem_C_T(annual_energy=1.7e3).values
    sc.prep.Q_dem_H_TH()
    p.Q_dem_H_TH.loc[:, "60/40"] = sc.prep.Q_dem_H_T(annual_energy=1.7e6).values

    # GRID
    sc.param("c_GRID_peak_", data=40, doc="Peak price", unit="€/kW_el")
    sc.param(
        "c_GRID_T", data=sc.prep.c_GRID_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el"
    )
    sc.prep.c_GRID_addon_T()
    sc.prep.ce_GRID_T()
    sc.var("E_GRID_buy_T", doc="Purchased electricity", unit="kWh_el")
    sc.var("E_GRID_sell_T", doc="Sold electricity", unit="kWh_el")
    sc.var("P_GRID_buyPeak_", doc="Peak electricity", unit="kW_el")

    # PV
    sc.param("P_PV_CAPx_", data=0, doc="Existing capacity", unit="kW_peak")
    sc.param("z_PV_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_PV_inv_())
    sc.param(from_db=db.k_PV_RMI_)
    sc.prep.E_PV_profile_T(use_coords=True)
    sc.var("E_PV_FI_T", doc="Feed-in", unit="kWh_el")
    sc.var("E_PV_OC_T", doc="Own consumption", unit="kWh_el")
    sc.var("E_PV_T", doc="Produced electricity", unit="kWh_el")
    sc.var("P_PV_CAPn_", doc="New capacity", unit="kW_el", ub=1e20 * p.z_PV_)

    # BES
    sc.param(from_db=db.eta_BES_in_)
    sc.param(from_db=db.eta_BES_time_)
    sc.param("k_BES_inPerCapa_", data=1.0, doc="Ratio charging power / capacity")
    sc.param("k_BES_outPerCapa_", data=1.0, doc="Ratio discharging power / capacity")
    sc.param("z_BES_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_BES_inv_(estimated_size=100, which="mean"))
    sc.param(from_db=db.k_BES_RMI_)
    sc.var("E_BES_CAPn_", doc="New capacity", unit="kWh_el")
    sc.var("E_BES_in_T", doc="Charged electricity", unit="kWh_el")
    sc.var("E_BES_inMax_", doc="Maximum charging rate", unit="kWh_el")
    sc.var("E_BES_out_T", doc="Discharged electricity", unit="kWh_el")
    sc.var("E_BES_outMax_", doc="Maximum discharging rate", unit="kWh_el")
    sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")

    # HP
    sc.param("T_C_C", data=273 + pd.Series([25, 35, 60], d.C), doc="Condensing temp.", unit="K")
    sc.param("T_H_in_H", data=273 + pd.Series([40, 70], d.H), doc="Heating inlet temp.", unit="K")
    sc.param("T_H_out_H", data=273 + pd.Series([60, 90], d.H), doc="Heating outlet temp.", unit="K")
    sc.param("T_N_in_N", data=273 + pd.Series([7, 30], d.N), doc="Cooling inlet temp.", unit="K")
    sc.param("T_N_out_N", data=273 + pd.Series([12, 35], d.N), doc="Cooling outlet temp.", unit="K")
    sc.param("eta_HP_", data=0.5, doc="Ratio of reaching the ideal COP")
    sc.param(
        "cop_HP_ideal_CN",
        data=pd.Series(
            {
                (c, n): (p.T_N_in_N[n] - 10) / ((p.T_C_C[c] + 10) - (p.T_N_in_N[n] - 10))
                for c in d.C
                for n in d.N
            }
        ),
    )
    sc.param(
        "cop_HP_CN",
        data=pd.Series(
            {
                (c, n): 0
                if c in ["25", "35"] and n == "30/35"
                else p.cop_HP_ideal_CN[c, n] * p.eta_HP_
                for c in d.C
                for n in d.N
            }
        ),
    )
    sc.param("P_HP_CAPx_N", data=[5000, 0], doc="Existing capacity", unit="kW_el")
    sc.param("P_HP_max_N", doc="Big-M number (upper bound for CAPn)", unit="kW_el", fill=5000)
    sc.param("z_HP_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_HP_inv_())
    sc.param(from_db=db.k_HP_RMI_)
    sc.var("E_HP_TCN", doc="Consumed electricity", unit="kWh_el")
    sc.var("P_HP_CAPn_N", doc="New capacity", unit="kW_el", ub=1e20 * p.z_HP_)
    sc.var("Q_HP_C_TCN", doc="Heat released on condensation side", unit="kWh_th")
    sc.var("Q_HP_E_TCN", doc="Heat absorbed on evaporation side", unit="kWh_th")
    sc.var("Y_HP_TCN", doc="1, If source and sink are connected at time-step", vtype=GRB.BINARY)

    # HOB
    sc.param("eta_HOB_", data=0.9, doc="Thermal efficiency", unit="kWh_th/kWh")
    sc.param(from_db=db.funcs.c_HOB_inv_())
    sc.param("Q_HOB_CAPx_", data=10000, doc="Existing capacity", unit="kW_th")
    sc.param(from_db=db.k_HOB_RMI_)
    sc.var("F_HOB_TF", doc="Consumed fuel", unit="kWh")
    sc.var("Q_HOB_CAPn_", doc="New capacity", unit="kW_th")
    sc.var("Q_HOB_T", doc="Produced heat", unit="kWh_th")

    # H2H1
    sc.var("Q_H2H1_T", doc="Heat down-grading", unit="kWh_th")

    # P2H
    sc.param("Q_P2H_CAPx_", data=10000, doc="Existing capacity", unit="kW_th")
    sc.var("E_P2H_T", doc="Consumed electricity", unit="kWh_el")
    sc.var("Q_P2H_T", doc="Produced heat", unit="kWh_th")

    # CHP
    sc.param(from_db=db.funcs.eta_CHP_th_(fuel="ng"))
    sc.param(from_db=db.funcs.eta_CHP_el_(fuel="ng"))
    sc.param("P_CHP_CAPx_", data=0, doc="Existing capacity", unit="kW_el")
    sc.param("z_CHP_", data=0, doc="If new capacity is allowed")
    sc.param("z_CHP_minPL_", data=0, doc="If minimal part load is modeled.")
    sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
    sc.param(from_db=db.k_CHP_RMI_)
    sc.var("E_CHP_FI_T", doc="Feed-in", unit="kWh_el")
    sc.var("E_CHP_OC_T", doc="Own consumption", unit="kWh_el")
    sc.var("E_CHP_T", doc="Produced electricity", unit="kWh_el")
    sc.var("F_CHP_TF", doc="Consumed fuel", unit="kWh")
    sc.var("P_CHP_CAPn_", doc="New capacity", unit="kW_el", ub=1e20 * p.z_CHP_)
    sc.var("Q_CHP_T", doc="Produced heat", unit="kWh_th")
    if p.z_CHP_minPL_:
        sc.par("k_CHP_minPL_", data=0.5, doc="Minimal allowed part load")
        sc.var("Y_CHP_T", doc="If in operation", vtype=GRB.BINARY)

    # HS
    sc.param("eta_HS_time_", data=0.995, doc="Storing efficiency")
    sc.param("k_HS_inPerCapa_", data=0.5, doc="Ratio loading power / capacity")
    sc.param("k_HS_outPerCapa_", data=0.5, doc="Ratio loading power / capacity")
    sc.param("z_HS_", data=0, doc="If new capacity is allowed")
    sc.param(from_db=db.funcs.c_HS_inv_(estimated_size=100, temp_spread=40))
    sc.param(from_db=db.k_HS_RMI_)
    sc.var("Q_HS_CAPn_H", doc="New capacity", unit="kWh_th", ub=1e20 * p.z_HS_)
    sc.var("Q_HS_in_TH", doc="Storage input", unit="kWh_th", lb=-GRB.INFINITY)
    sc.var("Q_HS_TH", doc="Storage level", unit="kWh_th")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T
    F = d.F
    N = d.N
    C = d.C
    H = d.H

    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    # C
    # Note: all energy-specific costs are divided by 1e3 to get k€ and therefore to
    # ensure smaller coefficient ranges which speeds up the optimization.
    m.addConstr(v.C_ == v.C_op_ + p.AF_ * v.C_inv_, "DEF_C_")
    m.addConstr(
        v.C_op_
        == (
            v.P_GRID_buyPeak_ * p.c_GRID_peak_ / 1e3
            + v.C_RMI_
            + p.n_comp_
            * quicksum(
                v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t]) / 1e3
                - v.E_GRID_sell_T[t] * p.c_GRID_T[t] / 1e3
                + quicksum((v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) * p.c_FUEL_F[f] / 1e3 for f in F)
                for t in T
            )
        ),
        "DEF_C_op_",
    )
    m.addConstr((v.C_inv_ == collectors.C_inv_(p, v) / 1e3))
    m.addConstr((v.C_RMI_ == collectors.C_RMI_(p, v) / 1e3), "DEF_C_RMI")

    # CE
    m.addConstr(
        v.CE_
        == quicksum(
            v.E_GRID_buy_T[t] * p.ce_GRID_T[t]
            + quicksum((v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) * p.ce_FUEL_F[f] for f in F)
            for t in T
        )
        / 1e6,
        "DEF_CE_",
    )

    # Electricity
    m.addConstrs(
        (
            v.E_GRID_buy_T[t] + v.E_CHP_OC_T[t] + v.E_PV_OC_T[t] + v.E_BES_out_T[t]
            == p.E_dem_T[t] + v.E_HP_TCN.sum(t, "*", "*") + v.E_P2H_T[t] + v.E_BES_in_T[t]
            for t in T
        ),
        "BAL_el",
    )

    # Heat
    m.addConstrs(
        (
            v.Q_HOB_T[t] + v.Q_CHP_T[t]
            == p.Q_dem_H_TH[t, "90/70"] + v.Q_H2H1_T[t] + v.Q_HS_in_TH[t, "90/70"]
            for t in T
        ),
        "BAL_H2",
    )

    # COOL
    m.addConstrs(
        (v.Q_HP_E_TCN.sum(t, "*", "7/12") == p.Q_dem_C_TN[t, "7/12"] for t in T), "BAL_HP_N1"
    )

    # GRID
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_CHP_FI_T[t] + v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_buyPeak_ for t in T), "DEF_peakPrice")

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs(
        (v.E_PV_T[t] == (p.P_PV_CAPx_ + v.P_PV_CAPn_) * p.E_PV_profile_T[t] for t in T), "PV1"
    )
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstr(v.E_BES_inMax_ == v.E_BES_CAPn_ * p.k_BES_inPerCapa_, "DEF_E_BES_inMax_")
    m.addConstr(v.E_BES_outMax_ == v.E_BES_CAPn_ * p.k_BES_outPerCapa_, "DEF_E_BES_outMax_")
    m.addConstrs((v.E_BES_in_T[t] <= v.E_BES_inMax_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_out_T[t] <= v.E_BES_outMax_ for t in T), "MAX_BES_OUT")
    m.addConstrs((v.E_BES_T[t] <= v.E_BES_CAPn_ for t in T), "MAX_BES_E")
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

    # CHP
    m.addConstrs(
        (v.E_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_el_ for f in F) for t in T), "CHP_E"
    )
    m.addConstrs(
        (v.Q_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_th_ for f in F) for t in T), "CHP_Q"
    )
    m.addConstrs((v.E_CHP_T[t] <= p.P_CHP_CAPx_ + v.P_CHP_CAPn_ for t in T), "CHP_CAP")
    m.addConstrs((v.E_CHP_T[t] == v.E_CHP_FI_T[t] + v.E_CHP_OC_T[t] for t in T), "CHP_OC_FI")

    # HOB
    m.addConstrs((v.Q_HOB_T[t] == v.F_HOB_TF.sum(t, "*") * p.eta_HOB_ for t in T), "BAL_HOB")
    m.addConstrs((v.Q_HOB_T[t] <= p.Q_HOB_CAPx_ + v.Q_HOB_CAPn_ for t in T), "CAP_HOB")

    # P2H
    m.addConstrs((v.Q_P2H_T[t] == v.E_P2H_T[t] for t in T), "BAL_P2H")
    m.addConstrs((v.Q_P2H_T[t] <= p.Q_P2H_CAPx_ for t in T), "CAP_P2H")

    # HP
    m.addConstrs(
        (
            v.Q_HP_E_TCN[t, c, n] == v.E_HP_TCN[t, c, n] * p.cop_HP_CN[c, n]
            for t in T
            for c in C
            for n in N
        ),
        "HP_BAL_1",
    )
    m.addConstrs(
        (
            v.Q_HP_C_TCN[t, c, n] == v.Q_HP_E_TCN[t, c, n] + v.E_HP_TCN[t, c, n]
            for t in T
            for c in C
            for n in N
        ),
        "HP_BAL_2",
    )
    m.addConstrs(
        (
            v.E_HP_TCN[t, c, n] <= v.Y_HP_TCN[t, c, n] * p.P_HP_max_N[n]
            for t in T
            for c in C
            for n in N
        ),
        "HP_BIGM",
    )
    m.addConstrs(
        (v.E_HP_TCN[t, c, n] <= v.P_HP_CAPn_N[n] for t in T for c in C for n in N), "CAP_HP"
    )
    m.addConstrs((v.Y_HP_TCN.sum(t, "*", "7/12") <= 1 for t in T), "HP_onlyOneConTemp1")
    m.addConstrs(
        (v.Y_HP_TCN.sum(t, c, "30/35") == 0 for t in T for c in ["25", "35"]), "HP_onlyOneConTemp2"
    )
    m.addConstrs((v.Y_HP_TCN[t, "60", "30/35"] == 1 for t in T), "HP_onlyOneConTemp3")
    m.addConstrs(
        (v.Q_HP_E_TCN.sum(t, "*", "30/35") == v.Q_HP_C_TCN[t, "35", "7/12"] for t in T), "BAL_HP_N2"
    )

    # HS
    m.addConstrs(
        (
            v.Q_HS_TH[t, h] == v.Q_HS_TH[t - 1, h] * p.eta_HS_time_ + v.Q_HS_in_TH[t - 1, h]
            for t in T[1:]
            for h in H
        ),
        "HS1",
    )
    m.addConstrs((v.Q_HS_TH[t, h] <= v.Q_HS_CAPn_H[h] for t in T for h in H), "HS3")
    m.addConstrs(
        (v.Q_HS_in_TH[t, h] <= v.Q_HS_CAPn_H[h] * p.k_HS_inPerCapa_ for t in T for h in H),
        "MAX_HS_IN",
    )
    m.addConstrs(
        (v.Q_HS_in_TH[t, h] >= -(v.Q_HS_CAPn_H[h] * p.k_HS_outPerCapa_) for t in T for h in H),
        "MAX_HS_OUT",
    )
    m.addConstrs((v.Q_HS_TH[t, h] == 0 for t in [min(T), max(T)] for h in H), "INI_HS")


def postprocess_func(r: draf.Results):
    r.make_pos_ent("E_GRID_buy_T")
    r.make_pos_ent("E_CHP_OC_T")
    r.make_pos_ent("Q_HS_in_TH", "Q_HS_out_TH")


def sankey_func(sc: draf.Scenario):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    F GAS CHP {r.F_CHP_TF.sum()}
    F GAS HOB {r.F_HOB_TF.sum()}
    E GRID EL {r.E_GRID_buy_T.sum()}
    E PV EL {r.E_PV_OC_T.sum()}
    E CHP EL {r.E_CHP_OC_T.sum()}
    E PV SELL_el {r.E_PV_FI_T.sum()}
    E CHP SELL_el {r.E_CHP_FI_T.sum()}
    E EL HP {r.E_HP_TCN.sum()}
    E EL DEM_el {p.E_dem_T.sum()}
    Q CHP H2 {r.Q_CHP_T.sum()}
    Q HOB H2 {r.Q_HOB_T.sum()}
    Q HP DEM_H1 {r.Q_HP_C_TCN[:,3,:].sum()}
    Q H2 H1 {r.Q_H2H1_T.sum()}
    Q H1 DEM_H1 {r.Q_H2H1_T.sum()}
    Q H2 DEM_H2 {p.Q_dem_H_TH[:,2].sum()}
    Q H1 SELL_th {r.Q_sell_TH[:,1].sum()}
    Q H2 SELL_th {r.Q_sell_TH[:,2].sum()}
    Q H1 LOSS_th {r.Q_HS_in_TH[:,1].sum() - r.Q_HS_out_TH[:,1].sum()}
    Q H2 LOSS_th {r.Q_HS_in_TH[:,2].sum() - r.Q_HS_out_TH[:,2].sum()}
    Q HOB LOSS_th {r.F_HOB_TF.sum() - r.Q_HOB_T.sum()}
    Q DEM_N1 HP {r.Q_HP_E_TCN[:,:,1].sum()}
    Q DEM_N2 HP {r.Q_HP_E_TCN[:,:,2].sum()}
    """


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
