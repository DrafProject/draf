"""DER-HUT: Distributed Energy Resources - Heat Upgrading Technologies"""

import pandas as pd
from gurobipy import GRB, Model, quicksum

import draf
from draf.prep import DataBase as db


def params_func(sc: draf.Scenario):

    # Alias
    p = sc.params
    d = sc.dims

    # Dimensions
    sc.dim("T", infer=True)
    sc.dim("F", ["ng", "bio"], doc="types of fuel")
    sc.dim("N", [1, 2], doc="cooling temperature level [1:7°C/12°C, 2:35°C/30°C]")
    sc.dim("C", [1, 2, 3], doc="condensing temperature levels [1:25°C, 2:35°C, 3:60°C]")
    sc.dim("H", [1, 2], doc="heating temperature levels [1: 40°C/60°C, 2: 70°C/90°]")

    # General
    sc.prep.n_comp_()
    sc.param("AF_", 0.1, doc="annuitiy factor (it pays off in 1/AF_ years)", unit="")
    sc.var("C_", doc="total costs", unit="€/a", lb=-GRB.INFINITY)
    sc.var("C_inv_", doc="investment costs", unit="€")
    sc.var("C_op_", doc="operating costs", unit="€/a", lb=-GRB.INFINITY)
    sc.var("CE_", doc="total emissions", unit="tCO2eq/a", lb=-GRB.INFINITY)
    sc.var("CE_op_", doc="operating emissions", unit="tCO2eq/a", lb=-GRB.INFINITY)

    # Pareto
    sc.param("alpha_", 0, doc="weighting factor", unit="")
    sc.param("n_C_", 1, doc="normalization factor", unit="")
    sc.param("n_CE_", 1 / 1e4, doc="normalization factor", unit="")

    # Fuels
    sc.param(from_db=db.c_FUEL_F)
    sc.param(from_db=db.ce_FUEL_F)

    # PV
    sc.param(from_db=db.funcs.c_PV_inv_())
    sc.prep.E_PV_profile_T()

    # Demands
    sc.prep.E_dem_T(profile="G1", annual_energy=4.8e6)
    sc.param("Q_dem_C_TN", doc="cooling demand", unit="kWh_th", fill=0)
    sc.param("Q_dem_H_TH", doc="heating demand", unit="kWh_th", fill=0)
    p.Q_dem_H_TH.loc[:, 1] = draf.prep.get_thermal_demand(
        ser_amb_temp=draf.prep.get_ambient_temp(2017, "60min"),
        annual_energy=1743000,
        target_temp=22,
        threshold_temp=13,
    )[d.T].values

    # GRID
    sc.param("c_GRID_peak_", 40, doc="peak price", unit="€/kW_el")
    sc.param("c_GRID_T", sc.prep.c_GRID_RTP_T())
    sc.var("E_GRID_buy_T", doc="purchased el.", unit="kWh_el")
    sc.var("E_GRID_sell_T", doc="sold el.", unit="kWh_el")
    sc.var("P_GRID_buy_peak_", doc="peak el.", unit="kW_el")
    sc.prep.c_GRID_addon_T()
    sc.prep.ce_GRID_T()

    # PV
    sc.param("P_PV_CAPx_", 0, doc="existing capacity", unit="kW_peak")
    sc.param("z_PV_", 0, doc="allowance for new capacity")
    sc.var("E_PV_FI_T", doc="feed-in", unit="kWh_el")
    sc.var("E_PV_OC_T", doc="own consumption", unit="kWh_el")
    sc.var("E_PV_T", doc="produced el.", unit="kWh_el")
    sc.var("P_PV_CAPn_", doc="new capacity", unit="kW_el", ub=1e20 * p.z_PV_)

    # BES
    sc.param("z_BES_", 0, doc="allowance for new capacity")
    sc.param(from_db=db.funcs.c_BES_inv_(estimated_size=100, which="mean"))
    sc.param("eta_BES_in_", 0.999, doc="loading efficiency", unit="")
    sc.param("eta_BES_time_", 0.999, doc="storing efficiency", unit="")
    sc.param("k_BES_in_per_capa_", 0.5, doc="ratio loading power / capacity", unit="")
    sc.var("E_BES_CAPn_", doc="new capacity", unit="kWh_el")
    sc.var("E_BES_in_max_", doc="maximum loading rate el.", unit="kWh_el")
    sc.var("E_BES_in_T", doc="loaded el.", unit="kWh_el", lb=-GRB.INFINITY)
    sc.var("E_BES_out_max_", doc="maximum unloading rate el.", unit="kWh_el")
    sc.var("E_BES_T", doc="el. stored", unit="kWh_el")

    # HP
    sc.param("z_HP_", 0, doc="allowance for new capacity")
    sc.param("T_N_in_N", 273 + pd.Series([7, 30], d.N), doc="temperature", unit="K")
    sc.param("T_N_out_N", 273 + pd.Series([12, 35], d.N), doc="temperature", unit="K")
    sc.param("T_C_C", 273 + pd.Series([25, 35, 60], d.C), doc="temperature", unit="K")
    sc.param("T_H_in_H", 273 + pd.Series([40, 70], d.H), doc="temperature", unit="K")
    sc.param("T_H_out_H", 273 + pd.Series([60, 90], d.H), doc="temperature", unit="K")
    sc.param(from_db=db.funcs.c_HP_inv_(estimated_size=100))
    sc.param("P_HP_CAPx_N", [5000, 0], doc="existing capacity", unit="kW_el")
    sc.param("P_HP_max_N", doc="Big-M number (upper bound for CAPn)", unit="kW_el", fill=5000)
    sc.param("eta_HP_", 0.5, doc="ratio of reaching the ideal COP")
    sc.param(
        "cop_HP_ideal_CN",
        pd.Series(
            {
                (c, n): (p.T_N_in_N[n] - 10) / ((p.T_C_C[c] + 10) - (p.T_N_in_N[n] - 10))
                for c in d.C
                for n in d.N
            }
        ),
    )
    sc.param(
        "cop_HP_CN",
        pd.Series(
            {
                (c, n): 0 if c in [1, 2] and n == 2 else p.cop_HP_ideal_CN[c, n] * p.eta_HP_
                for c in d.C
                for n in d.N
            }
        ),
    )
    sc.var("P_HP_CAPn_N", doc="new capacity", unit="kW_el", ub=1e20 * p.z_HP_)
    sc.var("Q_HP_E_TCN", doc="heat absorbed on evaporation side", unit="kWh_th")
    sc.var("Q_HP_C_TCN", doc="heat released on condensation side", unit="kWh_th")
    sc.var("E_HP_TCN", doc="consumed el.", unit="kWh_el")
    sc.var(
        "Y_HP_TCN",
        doc="1, if source and sink are connected at time-step",
        unit="",
        vtype=GRB.BINARY,
    )

    # HOB
    sc.param("Q_HOB_CAPx_", 10000, doc="existing capacity", unit="kW_th")
    sc.var("Q_HOB_CAPn_", doc="new capacity", unit="kW_th")
    sc.var("Q_HOB_T", doc="produced heat", unit="kWh_th")
    sc.var("F_HOB_TF", doc="consumed fuel", unit="kWh")

    # H2H1
    sc.var("Q_H2H1_T", doc="heat down-grading", unit="kWh_th")

    # P2H
    sc.var("Q_P2H_T", doc="produced heat", unit="kWh_th")
    sc.var("E_P2H_T", doc="consumed el.", unit="kWh_el")
    sc.param("Q_P2H_CAPx_", 10000, doc="existing capacity", unit="kW_th")

    # CHP
    sc.param("z_CHP_", 0, doc="allowance for new capacity")
    sc.param("P_CHP_CAPx_", 0, doc="existing capacity", unit="kW_el")
    sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
    sc.param("eta_CHP_el_", 0.35, doc="el. efficiency of CHP")
    sc.param("eta_CHP_th_", 0.98 - p.eta_CHP_el_, doc="thermal efficiency")
    sc.var("P_CHP_CAPn_", doc="new capacity", unit="kW_el", ub=1e20 * p.z_CHP_)
    sc.var("E_CHP_T", doc="produced el.", unit="kWh_el")
    sc.var("Q_CHP_T", doc="produced heat", unit="kWh_th")
    sc.var("F_CHP_TF", doc="consumed fuel", unit="kWh")
    sc.var("E_CHP_OC_T", doc="own consumption", unit="kWh_el")
    sc.var("E_CHP_FI_T", doc="feed-in", unit="kWh_el")

    # HS
    sc.param("z_HS_", 0, doc="allowance for new capacity")
    sc.param(from_db=db.funcs.c_HS_inv_(estimated_size=100, temp_spread=40))
    sc.param("eta_HS_time_", 0.995, doc="storing efficiency", unit="")
    sc.param("eta_HS_in_", 0.995, doc="loading efficiency", unit="")
    sc.param("k_HS_in_per_capa_", 0.5, doc="ratio loading power / capacity", unit="")
    sc.var("Q_HS_CAPn_H", doc="new capacity", unit="kWh_th", ub=1e20 * p.z_HS_)
    sc.var("Q_HS_TH", doc="storage level", unit="kWh_th")
    sc.var("Q_HS_in_TH", doc="storage input", unit="kWh_th", lb=-GRB.INFINITY)
    sc.var("Q_HS_in_max_")
    sc.var("Q_HS_out_max_")


def model_func(m: Model, d: draf.Dimensions, p: draf.Params, v: draf.Vars):
    T = d.T
    F = d.F
    N = d.N
    C = d.C
    H = d.H

    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    # Costs
    m.addConstr(v.C_ == v.C_op_ + p.AF_ * v.C_inv_, "DEF_C")
    m.addConstr(
        v.C_op_
        == (
            v.P_GRID_buy_peak_ * p.c_GRID_peak_
            + p.n_comp_
            * quicksum(
                v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t])
                - v.E_GRID_sell_T[t] * (p.c_GRID_T[t])
                + quicksum((v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) * p.c_FUEL_F[f] for f in F)
                for t in T
            )
        ),
        "DEF_C_op",
    )
    m.addConstr(
        v.C_inv_
        == (
            v.P_PV_CAPn_ * p.c_PV_inv_
            + v.P_CHP_CAPn_ * p.c_CHP_inv_
            + v.P_HP_CAPn_N.sum() * p.c_HP_inv_
            + v.Q_HS_CAPn_H.sum() * p.c_HS_inv_
            + v.E_BES_CAPn_ * p.c_BES_inv_
        ),
        "DEF_C_inv",
    )

    # Carbon Emissions
    m.addConstr(v.CE_ == v.CE_op_, "DEF_CE")
    m.addConstr(
        v.CE_op_
        == quicksum(
            v.E_GRID_buy_T[t] * p.ce_GRID_T[t]
            + quicksum((v.F_HOB_TF[t, f] + v.F_CHP_TF[t, f]) * p.ce_FUEL_F[f] for f in F)
            for t in T
        )
        / 1e6,
        "DEF_CE_op_",
    )

    # Electricity Grid
    m.addConstrs(
        (
            v.E_GRID_buy_T[t] + v.E_CHP_OC_T[t] + v.E_PV_OC_T[t]
            == p.E_dem_T[t] + v.E_HP_TCN.sum(t, "*", "*") + v.E_P2H_T[t] + v.E_BES_in_T[t]
            for t in T
        ),
        "BAL_el",
    )
    m.addConstrs((v.E_GRID_sell_T[t] == v.E_CHP_FI_T[t] + v.E_PV_FI_T[t] for t in T), "DEF_E_sell")
    m.addConstrs((v.E_GRID_buy_T[t] <= v.P_GRID_buy_peak_ for t in T), "DEF_peakPrice")

    # Heat
    m.addConstrs((v.Q_HP_E_TCN.sum(t, "*", 1) == float(p.Q_dem_C_TN[t, 1]) for t in T), "BAL_HP_N1")
    m.addConstrs(
        (
            v.Q_HOB_T[t] + v.Q_CHP_T[t] == p.Q_dem_H_TH[t, 2] + v.Q_H2H1_T[t] + v.Q_HS_in_TH[t, 2]
            for t in T
        ),
        "BAL_H2",
    )

    # TECHNOLOGIES =====================================

    # PV
    m.addConstrs(
        (v.E_PV_T[t] == (p.P_PV_CAPx_ + v.P_PV_CAPn_) * p.E_PV_profile_T[t] for t in T), "PV1"
    )
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

    # BES
    m.addConstrs(
        (
            v.E_BES_T[t] == p.eta_BES_time_ * v.E_BES_T[t - 1] + p.eta_BES_in_ * v.E_BES_in_T[t]
            for t in T[1:]
        ),
        "BAL_BES",
    )
    m.addConstrs((v.E_BES_T[t] <= v.E_BES_CAPn_ for t in T), "MAX_BES_E")
    m.addConstrs((v.E_BES_in_T[t] <= v.E_BES_in_max_ for t in T), "MAX_BES_IN")
    m.addConstrs((v.E_BES_in_T[t] >= -v.E_BES_out_max_ for t in T), "MAX_BES_OUT")
    m.addConstr(v.E_BES_in_max_ == v.E_BES_CAPn_ * p.k_BES_in_per_capa_, "DEF_E_BES_in_max_")
    m.addConstr(v.E_BES_out_max_ == v.E_BES_CAPn_ * p.k_BES_in_per_capa_, "DEF_E_BES_out_max_")
    m.addConstr(v.E_BES_T[min(T)] == 0, "INI_BES_0")
    m.addConstr(v.E_BES_T[max(T)] == 0, "END_BES_0")

    # CHP
    m.addConstrs(
        (v.E_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_el_ for f in F) for t in T), "CHP_E"
    )
    m.addConstrs(
        (v.Q_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_th_ for f in F) for t in T), "CHP_Q"
    )
    m.addConstrs((v.E_CHP_T[t] <= p.P_CHP_CAPx_ + v.P_CHP_CAPn_ for t in T), "CHP_CAP")
    m.addConstrs((v.E_CHP_T[t] == v.E_CHP_FI_T[t] + v.E_CHP_OC_T[t] for t in T), "CHP_OC_FI")
    m.addConstrs((v.Q_HOB_T[t] == v.F_HOB_TF.sum(t, "*") * 0.9 for t in T), "BAL_HOB")
    m.addConstrs((v.Q_HOB_T[t] <= p.Q_HOB_CAPx_ + v.Q_HOB_CAPn_ for t in T), "CAP_HOB")
    m.addConstrs((v.Q_P2H_T[t] == v.E_P2H_T[t] for t in T), "BAL_P2H")
    m.addConstrs((v.Q_P2H_T[t] <= p.Q_P2H_CAPx_ for t in T), "CAP_P2H")

    # HP
    m.addConstrs(
        (
            v.Q_HP_E_TCN[t, c, n] == p.cop_HP_CN[c, n] * v.E_HP_TCN[t, c, n]
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
    m.addConstrs((v.Y_HP_TCN.sum(t, "*", 1) <= 1 for t in T), "HP_onlyOneConTemp1")
    m.addConstrs((v.Y_HP_TCN.sum(t, c, 2) == 0 for t in T for c in [1, 2]), "HP_onlyOneConTemp2")
    m.addConstrs((v.Y_HP_TCN[t, 3, 2] == 1 for t in T), "HP_onlyOneConTemp3")
    m.addConstrs((v.Q_HP_E_TCN.sum(t, "*", 2) == v.Q_HP_C_TCN[t, 2, 1] for t in T), "BAL_HP_N2")

    # HS
    m.addConstrs(
        (
            v.Q_HS_TH[t, h]
            == p.eta_HS_time_ * v.Q_HS_TH[t - 1, h] + p.eta_HS_in_ * v.Q_HS_in_TH[t - 1, h]
            for t in T[1:]
            for h in H
        ),
        "HS1",
    )
    m.addConstrs((v.Q_HS_TH[t, h] <= v.Q_HS_CAPn_H[h] for t in T for h in H), "HS3")
    m.addConstrs((v.Q_HS_in_TH[t, h] <= v.Q_HS_in_max_ for t in T for h in H), "MAX_HS_IN")
    m.addConstrs((v.Q_HS_in_TH[t, h] >= -v.Q_HS_out_max_ for t in T for h in H), "MAX_HS_OUT")
    m.addConstrs(
        (v.Q_HS_in_max_ == v.Q_HS_CAPn_H[h] * p.k_HS_in_per_capa_ for h in H), "DEF_E_HS_in_max_"
    )
    m.addConstrs(
        (v.Q_HS_out_max_ == v.Q_HS_CAPn_H[h] * p.k_HS_in_per_capa_ for h in H), "DEF_E_HS_out_max_"
    )
    m.addConstrs((v.Q_HS_TH[min(T), h] == 0 for h in H), "INI_HS_0")
    m.addConstrs((v.Q_HS_TH[max(T), h] == 0 for h in H), "END_HS_0")


def postprocess_func(r: draf.Results):
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T")
    r.make_pos_ent("Q_HS_in_TH", "Q_HS_out_TH")
    r.make_pos_ent("E_CHP_OC_T")
    r.make_pos_ent("E_GRID_buy_T")


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
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min")
    cs.set_time_horizon(start="Jan-02 00", steps=24 * 2)
    cs.add_REF_scen().set_params(params_func)
    cs.add_scens(
        scen_vars=[("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3
    )
    cs.improve_pareto_and_set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs
