import numpy as np
import pandas as pd
from gurobipy import GRB, quicksum

import draf


def params_func(sc):
    p = sc.params
    T = sc.add_dim("T", infer=True)
    F = sc.add_dim("F", ["ng", "bio"], doc="types of fuel")
    E = sc.add_dim("E", ["priv", "busi"], doc="types of electric vehicle")
    L = sc.add_dim("L", [1, 2], doc="waste heat streams")
    N = sc.add_dim("N", [1, 2], doc="cooling temperature level [1:7°C/12°C, 2:35°C/30°C]")
    C = sc.add_dim("C", [1, 2, 3], doc="condensing temperature levels [1:25°C, 2:35°C, 3:60°C]")
    H = sc.add_dim("H", [1, 2], doc="heating temperature levels [1: 40°C/60°C, 2: 70°C/90°]")

    sc.add_par("alpha_", 0, doc="weighting factor", unit="")
    sc.add_par(
        "n_c_", 8760 / len(T), doc="cost weighting factor to compensate part year analysis", unit=""
    )
    sc.add_par("n_C_", 1, doc="normalization factor", unit="")
    sc.add_par("n_CE_", 1 / 1e4, doc="normalization factor", unit="")
    sc.add_par("AF_", 0.1, doc="annuitiy factor (it pays off in 1/AF_ years)", unit="")

    doc_CAPx = "existing capacity"
    sc.add_par("P_HP_CAPx_N", [5000, 0], doc=doc_CAPx, unit="kW_el")
    sc.add_par("P_CHP_CAPx_", 0, doc=doc_CAPx, unit="kW_el")
    sc.add_par("P_PV_CAPx_", 0, doc=doc_CAPx, unit="kW_peak")
    sc.add_par("P_BEV_CAPx_E", doc=doc_CAPx, unit="kWh_el", fill=18 * 38)
    sc.add_par("Q_BOI_CAPx_", 10000, doc=doc_CAPx, unit="kW_th")
    sc.add_par("Q_P2H_CAPx_", 10000, doc=doc_CAPx, unit="kW_th")

    # bounds for new capacities
    sc.add_par("P_HP_max_N", doc="Big-M number (upper bound for CAPn)", unit="kW_el", fill=5000)

    doc_z = "allowance for new capacity"
    sc.add_par("z_PV_", 0, doc=doc_z)
    sc.add_par("z_HP_", 0, doc=doc_z)
    sc.add_par("z_HS_", 0, doc=doc_z)
    sc.add_par("z_CHP_", 0, doc=doc_z)
    sc.add_par("z_BES_", 0, doc=doc_z)

    # COSTS
    sc.add_par("c_th_F", [0.04, 0.04], doc="fuel cost", unit="€/kWh")
    sc.add_par("c_el_peak_", 40, doc="peak price", unit="€/kW_el")
    sc.prep.add_c_GRID_addon_T()

    sc.add_par(
        "c_PV_inv_",
        500,
        doc="invest cost, valid for: (??), source: short google search todo:validate",
        unit="€/kW_peak",
    )
    sc.add_par(
        "c_HP_inv_",
        200 * 4,
        doc="invest cost, valid for: (100 kW_th), source: Wolf(2016)",
        unit="€/kW_el",
    )
    sc.add_par(
        "c_HS_inv_",
        30,
        doc="invest cost, valid for: (>100kWh, 40Kelvin Delta, 60°C spez. Kapa), source: ffe(2017)",
        unit="€/kWh_th",
    )
    sc.add_par(
        "c_BES_inv_",
        100,
        doc="invest cost, valid for: (year 2020), source:  Horváth & Partners, Statista(2018),",
        unit="€/kWh_el_inst",
    )

    sc.add_par(
        "c_CHP_inv_",
        1200,
        doc="invest cost, valid for: (~100 kW_el), source: Armin Bez(2012)",
        unit="€/kW_el",
    )
    # sc.add_par("c_CHP_inv_", 500, "invest cost, valid for: (~400 kW_el), source: Armin Bez(2012)", "€/kW_el")
    # sc.add_par("c_CHP_inv_", 400, "invest cost, valid for: (~1000 kW_el), source: Armin Bez(2012)", "€/kW_el")
    # sc.add_par("c_CHP_inv_", 250, "invest cost, valid for: (~2000 kW_el), source: Armin Bez(2012)", "€/kW_el")

    # ENVIRONMENT
    sc.prep.add_ce_GRID_T()
    sc.add_par(
        "ce_th_F",
        [202, 71],
        doc="fuel carbon emissions, source: https://lfu.brandenburg.de/cms/detail.php/bb1.c.523833.de",
        unit="gCO2eq/kWh_el",
    )

    # TEMPERATURES
    offset = np.array(273)
    sc.add_par("T_N_in_N", offset + pd.Series([7, 30], N), doc="temperature", unit="K")
    sc.add_par("T_N_out_N", offset + pd.Series([12, 35], N), doc="temperature", unit="K")
    sc.add_par("T_C_C", offset + pd.Series([25, 35, 60], C), doc="temperature", unit="K")
    sc.add_par("T_H_in_H", offset + pd.Series([40, 70], H), doc="temperature", unit="K")
    sc.add_par("T_H_out_H", offset + pd.Series([60, 90], H), doc="temperature", unit="K")

    # ENVIRONMENT
    sc.prep.add_E_PV_profile_T()

    # DEMANDS
    sc.prep.add_E_dem_T(profile="G1", annual_energy=4.8e6)
    sc.add_par("Q_dem_C_TN", doc="cooling demand", unit="kWh_th", fill=0)
    sc.add_par("Q_dem_H_TH", doc="heating demand", unit="kWh_th", fill=0)
    p.Q_dem_H_TH.loc[:, 1] = draf.io.get_thermal_demand(
        ser_amb_temp=draf.io.get_ambient_temp(2017, "60min"),
        annual_energy=1743000,
        target_temp=22,
        threshold_temp=13,
    )[T].values

    # EFFICIENCIES
    sc.add_par("eta_HP_", 0.5, doc="ratio of reaching the ideal COP")
    z = sc.add_par("eta_CHP_el_", 0.35, doc="el. efficiency of CHP")
    sc.add_par("eta_CHP_th_", 0.98 - z, doc="thermal efficiency")

    sc.add_par(
        "cop_HP_ideal_CN",
        pd.Series(
            {
                (c, n): (p.T_N_in_N[n] - 10) / ((p.T_C_C[c] + 10) - (p.T_N_in_N[n] - 10))
                for c in C
                for n in N
            }
        ),
    )
    sc.add_par(
        "cop_HP_CN",
        pd.Series(
            {
                (c, n): 0 if c in [1, 2] and n == 2 else p.cop_HP_ideal_CN[c, n] * p.eta_HP_
                for c in C
                for n in N
            }
        ),
    )
    sc.add_par("eta_HS_time_", 0.995, doc="storing efficiency", unit="")
    sc.add_par("eta_HS_in_", 0.995, doc="loading efficiency", unit="")
    sc.add_par("k_HS_in_per_capa_", 0.5, doc="ratio loading power / capacity", unit="")

    sc.add_par("eta_BES_time_", 0.999, doc="storing efficiency", unit="")
    sc.add_par("eta_BES_in_", 0.999, doc="loading efficiency", unit="")
    sc.add_par("k_BES_in_per_capa_", 0.5, doc="ratio loading power / capacity", unit="")

    ###### VARIABLES ######

    sc.add_var("C_", doc="total costs", unit="€/a", lb=-GRB.INFINITY)
    sc.add_var("C_inv_", doc="investment costs", unit="€")
    sc.add_var("C_op_", doc="operating costs", unit="€/a", lb=-GRB.INFINITY)
    sc.add_var("CE_", doc="total emissions", unit="tCO2eq/a", lb=-GRB.INFINITY)
    sc.add_var("CE_op_", doc="operating emissions", unit="tCO2eq/a", lb=-GRB.INFINITY)

    sc.add_var("E_GRID_buy_T", doc="purchased el.", unit="kWh_el")
    sc.add_var("E_GRID_sell_T", doc="purchased el.", unit="kWh_el")
    sc.add_var("P_GRID_buy_peak_", doc="peak el.", unit="kW_el")

    sc.add_var("P_PV_CAPn_", doc="new capacity", unit="kW_el", ub=1e20 * p.z_PV_)
    sc.add_var("E_PV_T", doc="produced el.", unit="kWh_el")
    sc.add_var("E_PV_OC_T", doc="own consumption", unit="kWh_el")
    sc.add_var("E_PV_FI_T", doc="feed-in", unit="kWh_el")

    sc.add_var("E_BES_CAPn_", doc="new capacity", unit="kWh_el")
    sc.add_var("E_BES_T", doc="el. stored", unit="kWh_el")
    sc.add_var("E_BES_in_T", doc="loaded el.", unit="kWh_el", lb=-GRB.INFINITY)
    sc.add_var("E_BES_in_max_", doc="maximum loading rate el.", unit="kWh_el")
    sc.add_var("E_BES_out_max_", doc="maximum unloading rate el.", unit="kWh_el")

    sc.add_var("P_HP_CAPn_N", doc="new capacity", unit="kW_el", ub=1e20 * p.z_HP_)
    sc.add_var("Q_HP_E_TCN", doc="heat absorbed on evaporation side", unit="kWh_th")
    sc.add_var("Q_HP_C_TCN", doc="heat released on condensation side", unit="kWh_th")
    sc.add_var("E_HP_TCN", doc="consumed el.", unit="kWh_el")
    sc.add_var(
        "Y_HP_TCN",
        doc="1, if source and sink are connected at time-step",
        unit="",
        vtype=GRB.BINARY,
    )

    sc.add_var("Q_BOI_CAPn_", doc="new capacity", unit="kW_th")
    sc.add_var("Q_BOI_T", doc="produced heat", unit="kWh_th")
    sc.add_var("F_BOI_TF", doc="consumed fuel", unit="kWh")

    sc.add_var("Q_H2H1_T", doc="heat down-grading", unit="kWh_th")
    sc.add_var("Q_sell_TH", doc="sold heat", unit="kWh_th")
    sc.add_var("Q_P2H_T", doc="produced heat", unit="kWh_th")
    sc.add_var("E_P2H_T", doc="consumed el.", unit="kWh_el")

    sc.add_var("P_CHP_CAPn_", doc="new capacity", unit="kW_el", ub=1e20 * p.z_CHP_)
    sc.add_var("E_CHP_T", doc="produced el.", unit="kWh_el")
    sc.add_var("Q_CHP_T", doc="produced heat", unit="kWh_th")
    sc.add_var("F_CHP_TF", doc="consumed fuel", unit="kWh")
    sc.add_var("E_CHP_OC_T", doc="own consumption", unit="kWh_el")
    sc.add_var("E_CHP_FI_T", doc="feed-in", unit="kWh_el")

    sc.add_var("Q_HS_CAPn_H", doc="new capacity", unit="kWh_th", ub=1e20 * p.z_HS_)
    sc.add_var("Q_HS_TH", doc="storage level", unit="kWh_th")
    sc.add_var("Q_HS_in_TH", doc="storage input", unit="kWh_th", lb=-GRB.INFINITY)
    sc.add_var("Q_HS_in_max_")
    sc.add_var("Q_HS_out_max_")

    sc.add_var("Q_WH_TL", doc="absorbed waste-heat", unit="kWh_th")
    sc.add_var("A_WH_EX_L", doc="area of heat exchanger", unit="m^2")

    sc.prep.add_c_GRID_RTP_T()
    sc.add_par("c_GRID_T", sc.params.c_GRID_RTP_T)


def model_func(m, d, p, v):
    T = d.T
    F = d.F
    L = d.L
    N = d.N
    C = d.C
    H = d.H
    E = d.E

    m.setObjective(((1 - p.alpha_) * v.C_ * p.n_C_ + p.alpha_ * v.CE_ * p.n_CE_), GRB.MINIMIZE)

    m.addConstr(v.C_ == v.C_op_ + p.AF_ * v.C_inv_, "DEF_C")

    m.addConstr(
        v.C_op_
        == (
            v.P_GRID_buy_peak_ * p.c_el_peak_
            + p.n_c_
            * quicksum(
                v.E_GRID_buy_T[t] * (p.c_GRID_T[t] + p.c_GRID_addon_T[t])
                - v.E_GRID_sell_T[t] * (p.c_GRID_T[t])
                + quicksum((v.F_BOI_TF[t, f] + v.F_CHP_TF[t, f]) * p.c_th_F[f] for f in F)
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

    m.addConstr(v.CE_ == v.CE_op_, "DEF_CE")

    m.addConstr(
        v.CE_op_
        == quicksum(
            v.E_GRID_buy_T[t] * p.ce_GRID_T[t]
            + quicksum((v.F_BOI_TF[t, f] + v.F_CHP_TF[t, f]) * p.ce_th_F[f] for f in F)
            for t in T
        )
        / 1e6,
        "DEF_CE_op_",
    )

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

    m.addConstrs(
        (v.E_PV_T[t] == (p.P_PV_CAPx_ + v.P_PV_CAPn_) * p.E_PV_profile_T[t] for t in T), "PV1"
    )
    m.addConstrs((v.E_PV_T[t] == v.E_PV_FI_T[t] + v.E_PV_OC_T[t] for t in T), "PV_OC_FI")

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

    # m.addConstrs((v.E_BEV_TE[t, e] == p.eta_BEV_time_ * v.E_BEV_TE[t - 1, e] + p.eta_BEV_in_ * v.E_BEV_in_TE[t, e] - v.E_BEV_out_drive_TE[t, e] - v.E_BEV_out_V2X_TE[t, e] for t in T[1:] for e in E), "BAL_BEV")
    # m.addConstrs((v.E_BEV_TE[t, e] <= p.P_BEV_CAPx_E[e] for t in T for e in E), "MAX_BEV_E")
    # m.addConstrs((v.E_BEV_in_TE[t, e] <= p.y_BEV_avail_TE[t, e] * p.P_BEV_CAPx_E[e] * p.k_BEV_in_per_capa_ for t in T for e in E), "MAX_BEV_IN")
    # m.addConstrs((v.E_BEV_out_V2X_TE[t, e] <= p.z_BEV_V2X_ * p.y_BEV_avail_TE[t, e] * p.k_BEV_out_per_capa_ * p.P_BEV_CAPx_E[e] for t in T for e in E), "MAX_BEV_OUT_use")
    # m.addConstrs((v.E_BEV_out_drive_TE[t,e] == 0 for t in T for e in E if p.y_BEV_avail_TE[t, e] == 1), "RES_BEV")
    # m.addConstrs((v.E_BEV_TE[t, e] == p.k_BEV_empty_E[e] * p.P_BEV_CAPx_E[e] for t in T[1:] for e in E if p.y_BEV_empty_TE[t, e]), "INI_BEV_empty")
    # m.addConstrs((v.E_BEV_TE[t, e] == p.k_BEV_full_E[e] * p.P_BEV_CAPx_E[e] for t in T[1:] for e in E if p.y_BEV_full_TE[t, e]), "INI_BEV_full")

    m.addConstrs(
        (v.E_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_el_ for f in F) for t in T), "CHP_E"
    )
    m.addConstrs(
        (v.Q_CHP_T[t] == quicksum(v.F_CHP_TF[t, f] * p.eta_CHP_th_ for f in F) for t in T), "CHP_Q"
    )
    m.addConstrs((v.E_CHP_T[t] <= p.P_CHP_CAPx_ + v.P_CHP_CAPn_ for t in T), "CHP_CAP")
    m.addConstrs((v.E_CHP_T[t] == v.E_CHP_FI_T[t] + v.E_CHP_OC_T[t] for t in T), "CHP_OC_FI")

    m.addConstrs((v.Q_BOI_T[t] == v.F_BOI_TF.sum(t, "*") * 0.9 for t in T), "BAL_BOI")
    m.addConstrs((v.Q_BOI_T[t] <= p.Q_BOI_CAPx_ + v.Q_BOI_CAPn_ for t in T), "CAP_BOI")

    m.addConstrs((v.Q_P2H_T[t] == v.E_P2H_T[t] for t in T), "BAL_P2H")
    m.addConstrs((v.Q_P2H_T[t] <= p.Q_P2H_CAPx_ for t in T), "CAP_P2H")

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
    m.addConstrs((v.Q_HP_E_TCN.sum(t, "*", 1) == float(p.Q_dem_C_TN[t, 1]) for t in T), "BAL_HP_N1")
    m.addConstrs((v.Q_HP_E_TCN.sum(t, "*", 2) == v.Q_HP_C_TCN[t, 2, 1] for t in T), "BAL_HP_N2")

    m.addConstrs(
        (
            v.Q_BOI_T[t] + v.Q_CHP_T[t] + v.Q_WH_TL.sum(t, "*")
            == p.Q_dem_H_TH[t, 2] + v.Q_H2H1_T[t] + v.Q_HS_in_TH[t, 2]
            for t in T
        ),
        "BAL_H2",
    )

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

    # m.addConstrs((v.Q_WH_TL[t, l] <= p.m_WH_TL[t, l]
    #               * p.c_WH_L[l] * p.dt_WH_L[l] for t in T for l in L), "WH1")
    # m.addConstrs((v.Q_WH_TL[t, l] <= p.U_WH_L[l] * p.dt_WH_2_L[l]
    #               * v.A_WH_EX_L[l] for t in T for l in L), "WH2")


def postprocess_func(r):
    r.make_pos_ent("E_BES_in_T", "E_BES_out_T")
    r.make_pos_ent("Q_HS_in_TH", "Q_HS_out_TH")
    r.make_pos_ent("E_CHP_OC_T")


def sankey_func(sc):
    p = sc.params
    r = sc.res
    return f"""\
    type source target value
    F GAS CHP {r.F_CHP_TF.sum()}
    F GAS BOI {r.F_BOI_TF.sum()}
    E PUR EL {r.E_GRID_buy_T.sum()}
    E PV EL {r.E_PV_OC_T.sum()}
    E CHP EL {r.E_CHP_OC_T.sum()}
    E PV SELL_el {r.E_PV_FI_T.sum()}
    E CHP SELL_el {r.E_CHP_FI_T.sum()}
    E EL HP {r.E_HP_TCN.sum()}
    E EL DEM_el {p.E_dem_T.sum()}
    Q CHP H2 {r.Q_CHP_T.sum()}
    Q BOI H2 {r.Q_BOI_T.sum()}
    Q HP DEM_H1 {r.Q_HP_C_TCN[:,3,:].sum()}
    Q H2 H1 {r.Q_H2H1_T.sum()}
    Q H1 DEM_H1 {r.Q_H2H1_T.sum()}
    Q H2 DEM_H2 {p.Q_dem_H_TH[:,2].sum()}
    Q H1 SELL_th {r.Q_sell_TH[:,1].sum()}
    Q H2 SELL_th {r.Q_sell_TH[:,2].sum()}
    Q H1 LOSS_th {r.Q_HS_in_TH[:,1].sum() - r.Q_HS_out_TH[:,1].sum()}
    Q H2 LOSS_th {r.Q_HS_in_TH[:,2].sum() - r.Q_HS_out_TH[:,2].sum()}
    Q BOI LOSS_th {r.F_BOI_TF.sum() - r.Q_BOI_T.sum()}
    Q DEM_N1 HP {r.Q_HP_E_TCN[:,:,1].sum()}
    Q DEM_N2 HP {r.Q_HP_E_TCN[:,:,2].sum()}
    """


def main():
    cs = draf.CaseStudy("DER_HUT", year=2019, freq="60min")
    cs.set_datetime_filter(start="Jan-02 00", steps=24 * 2)
    sc = cs.add_REF_scen()
    sc.set_params(params_func)
    cs.add_scens(
        scen_vars=[("c_GRID_T", "t", [f"c_GRID_{ix}_T" for ix in ["RTP"]])], nParetoPoints=3
    )
    cs.improve_pareto_and_set_model(model_func)
    cs.optimize(logToConsole=False, postprocess_func=postprocess_func)
    return cs


if __name__ == "__main__":
    cs = main()
