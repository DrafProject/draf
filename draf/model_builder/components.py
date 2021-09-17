import logging
from dataclasses import dataclass

import pandas as pd
from gurobipy import GRB, Model, quicksum

from draf import Dimensions, Params, Results, Scenario, Vars
from draf.helper import set_component_order_by_dependency
from draf.model_builder import collectors
from draf.model_builder.abstract_component import Component
from draf.prep import DataBase as db

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


@dataclass
class PRE(Component):
    """The base component including T, General"""

    def param_func(self, sc: Scenario):
        sc.balance("P_EL_source_T", doc="Power sources", unit="kW_el")
        sc.balance("P_EL_sink_T", doc="Power sinks", unit="kW_el")
        sc.balance("P_EG_sell_T", doc="Sold electricity power", unit="kW_el")
        sc.balance("dQ_source_TL", doc="Thermal energy flow sources", unit="kW_th")
        sc.balance("dQ_sink_TL", doc="Thermal energy flow sinks", unit="kW_th")
        sc.balance("C_TOT_", doc="Total costs", unit="k€/a")
        sc.balance("C_TOT_op_", doc="Total operating costs", unit="k€/a")
        sc.balance("CE_TOT_", doc="Total carbon emissions", unit="kgCO2eq/a")
        sc.balance("Penalty_", doc="Penalty term for objective function", unit="")

        # General
        sc.dim("T", infer=True)
        sc.prep.k__comp_()
        sc.prep.k__dT_()

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        pass


@dataclass
class POST(Component):
    """The base component including T, General, Total, Pareto"""

    def param_func(self, sc: Scenario):
        # Total
        sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)
        if sc.consider_invest:
            sc.param("k__AF_", 0.1, doc="Annuity factor (it pays off in 1/k__AF_ years)")
            sc.var("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.var("C_TOT_RMI_", doc="Total annual maintainance cost", unit="k€")
        sc.var("Penalty_", "Penalty term for objective function")

        # Pareto
        sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
        sc.param("k_PTO_C_", data=1, doc="Normalization factor")
        sc.param("k_PTO_CE_", data=1 / 1e4, doc="Normalization factor")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):

        m.setObjective(
            (
                (1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_
                + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_
                + v.Penalty_
            ),
            GRB.MINIMIZE,
        )

        # C
        m.addConstr(v.C_TOT_op_ == quicksum(sc.balances.C_TOT_op_.values()), "DEF_C_TOT_op_")
        sc.balances.C_TOT_["op"] = v.C_TOT_op_

        if sc.consider_invest:
            m.addConstr((v.C_TOT_inv_ == collectors.C_TOT_inv_(p, v) / 1e3))
            m.addConstr((v.C_TOT_RMI_ == collectors.C_TOT_RMI_(p, v) / 1e3), "DEF_C_TOT_RMI_")
            sc.balances.C_TOT_["RMI"] = v.C_TOT_RMI_
            sc.balances.C_TOT_["inv"] = p.k__AF_ * v.C_TOT_inv_

        m.addConstr(v.C_TOT_ == quicksum(sc.balances.C_TOT_.values()), "DEF_C_TOT_")

        # CE
        m.addConstr(v.CE_TOT_ == quicksum(sc.balances.CE_TOT_.values()), "DEF_CE_TOT_")

        # Penalty
        m.addConstr(v.Penalty_ == quicksum(sc.balances.Penalty_.values()), "DEF_Penalty_")

        # Electricity
        m.addConstrs(
            (
                quicksum(x(t) for x in sc.balances.P_EL_source_T.values())
                == quicksum(x(t) for x in sc.balances.P_EL_sink_T.values())
                for t in d.T
            ),
            "BAL_el",
        )

        # Thermal energy
        if sc.has_thermal_entities:
            m.addConstrs(
                (
                    quicksum(x(t, l) for x in sc.balances.dQ_source_TL.values())
                    == quicksum(x(t, l) for x in sc.balances.dQ_sink_TL.values())
                    for t in d.T
                    for l in d.L
                ),
                "BAL_th",
            )


@dataclass
class cDem(Component):
    """Cooling demand"""

    def param_func(self, sc: Scenario):
        sc.dim("N", ["7/12", "30/35"], doc="Cooling temperature levels (inlet / outlet) in °C")

        sc.prep.dQ_cDem_TN()
        sc.params.dQ_cDem_TN.loc[:, "7/12"] = sc.prep.dQ_cDem_T(annual_energy=1.7e3).values
        sc.param(
            "T_cDem_in_N",
            data=273 + pd.Series([7, 30], sc.dims.N),
            doc="Cooling inlet temp.",
            unit="K",
        )
        sc.param(
            "T_cDem_out_N",
            data=273 + pd.Series([12, 35], sc.dims.N),
            doc="Cooling outlet temp.",
            unit="K",
        )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        sc.balances.dQ_source_TL["cDem"] = lambda t, l: p.dQ_cDem_TN[t, l] if l in d.N else 0


@dataclass
class hDem(Component):
    """Heating demand"""

    def param_func(self, sc: Scenario):
        sc.dim("H", ["60/40", "90/70"], doc="Heating temperature levels (inlet / outlet) in °C")

        sc.prep.dQ_hDem_TH()
        sc.params.dQ_hDem_TH.loc[:, "60/40"] = sc.prep.dQ_hDem_T(annual_energy=1.7e6).values
        sc.param(
            "T_hDem_out_H",
            data=273 + pd.Series([40, 70], sc.dims.H),
            doc="Heating outlet temp.",
            unit="K",
        )
        sc.param(
            "T_hDem_in_H",
            data=273 + pd.Series([60, 90], sc.dims.H),
            doc="Heating inlet temp.",
            unit="K",
        )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        sc.balances.dQ_sink_TL["hDem"] = lambda t, l: p.dQ_hDem_TH[t, l] if l in d.H else 0


@dataclass
class eDem(Component):
    """Electricity demand"""

    profile: str = "G3"
    annual_energy: float = 5e6

    def param_func(self, sc: Scenario):
        sc.prep.P_eDem_T(profile=self.profile, annual_energy=self.annual_energy)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        sc.balances.P_EL_sink_T["eDem"] = lambda t: p.P_eDem_T[t]


@dataclass
class EG(Component):
    """Electricity grid"""

    c_buyPeak: float = 50.0

    def param_func(self, sc: Scenario):
        sc.param("c_EG_buyPeak_", data=self.c_buyPeak, doc="Peak price", unit="€/kW_el")
        sc.param(
            "c_EG_T", data=sc.prep.c_EG_RTP_T(), doc="Chosen electricity tariff", unit="€/kWh_el"
        )
        sc.prep.c_EG_TOU_T()
        sc.prep.c_EG_FLAT_T()
        sc.prep.c_EG_addon_T()
        sc.prep.ce_EG_T()
        sc.var("P_EG_buy_T", doc="Purchased electrical power", unit="kW_el")
        sc.var("P_EG_sell_T", doc="Selling electrical power", unit="kW_el")
        sc.var("P_EG_buyPeak_", doc="Peak electrical power", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        m.addConstrs(
            (v.P_EG_sell_T[t] == sum(x(t) for x in sc.balances.P_EG_sell_T.values()) for t in d.T),
            "DEF_E_sell",
        )
        m.addConstrs((v.P_EG_buy_T[t] <= v.P_EG_buyPeak_ for t in d.T), "DEF_peakPrice")

        sc.balances.P_EL_source_T["EL"] = lambda t: v.P_EG_buy_T[t]
        sc.balances.P_EL_sink_T["EL"] = lambda t: v.P_EG_sell_T[t]
        sc.balances.C_TOT_op_["EG_peak"] = v.P_EG_buyPeak_ * p.c_EG_buyPeak_ / 1e3
        sc.balances.C_TOT_op_["EG"] = (
            p.k__comp_
            * p.k__dT_
            / 1e3
            * quicksum(
                v.P_EG_buy_T[t] * (p.c_EG_T[t] + p.c_EG_addon_T[t]) - v.P_EG_sell_T[t] * p.c_EG_T[t]
                for t in d.T
            )
        )
        sc.balances.CE_TOT_["EG"] = (
            p.k__comp_
            * p.k__dT_
            * quicksum(p.ce_EG_T[t] * (v.P_EG_buy_T[t] - v.P_EG_sell_T[t]) for t in d.T)
        )

    def postprocess_func(self, r: Results):
        r.make_pos_ent("P_EG_buy_T")


@dataclass
class Fuel(Component):
    """Fuels"""

    c_ceTax: float = 55

    def param_func(self, sc: Scenario):
        sc.dim("F", ["ng", "bio"], doc="Types of fuel")
        sc.balance("F_fuel_F", doc="Fuel power", unit="kW")
        sc.param(from_db=db.c_Fuel_F)
        sc.param("c_Fuel_ceTax_", data=self.c_ceTax, doc="Carbon tax on fuel", unit="€/tCO2eq")
        sc.param(from_db=db.ce_Fuel_F)
        sc.var("C_Fuel_ceTax_", doc="Carbon tax on fuel", unit="€/a")
        sc.var("CE_Fuel_", doc="Total carbon emissions for fuel", unit="kgCO2eq/a")
        sc.var("F_fuel_F", doc="Total fuel consumption", unit="kW")

    def model_func(self, sc, m, d, p, v):
        m.addConstrs(
            (v.F_fuel_F[f] == quicksum(x(f) for x in sc.balances.F_fuel_F.values()) for f in d.F),
            "DEF_F_fuel_F",
        )
        m.addConstr(
            v.CE_Fuel_
            == p.k__comp_ * p.k__dT_ * quicksum(v.F_fuel_F[f] * p.ce_Fuel_F[f] for f in d.F)
        )
        m.addConstr(v.C_Fuel_ceTax_ == p.c_Fuel_ceTax_ * v.CE_Fuel_)
        sc.balances.CE_TOT_["Fuel"] = v.CE_Fuel_
        sc.balances.C_TOT_op_["Fuel"] = (
            p.k__comp_ * p.k__dT_ * quicksum(v.F_fuel_F[f] * p.c_Fuel_F[f] / 1e3 for f in d.F)
            + v.C_Fuel_ceTax_
        )


@dataclass
class BES(Component):
    """Battery Energy Storage"""

    E_CAPx: float = 100
    allow_new: bool = False

    def param_func(self, sc: Scenario):
        sc.param("E_BES_CAPx_", data=self.E_CAPx, doc="Existing capacity", unit="kWh_el")
        sc.param("k_BES_ini_", data=0, doc="Initial and final energy filling share")
        sc.param(from_db=db.eta_BES_cycle_)
        sc.param(from_db=db.eta_BES_time_)
        sc.param(from_db=db.k_BES_inPerCap_)
        sc.param(from_db=db.k_BES_outPerCap_)
        sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")
        sc.var("P_BES_in_T", doc="Charging power", unit="kW_el")
        sc.var("P_BES_out_T", doc="Discharging power", unit="kW_el")

        if sc.consider_invest:
            sc.param(from_db=db.k_BES_RMI_)
            sc.param("z_BES_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_BES_inv_(estimated_size=100, which="mean"))
            sc.var("E_BES_CAPn_", doc="New capacity", unit="kWh_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        cap = p.E_BES_CAPx_ + v.E_BES_CAPn_ if sc.consider_invest else p.E_BES_CAPx_
        m.addConstrs(
            (v.P_BES_in_T[t] <= p.k_BES_inPerCap_ * cap for t in d.T),
            "MAX_BES_IN",
        )
        m.addConstrs(
            (v.P_BES_out_T[t] <= p.k_BES_outPerCap_ * cap for t in d.T),
            "MAX_BES_OUT",
        )
        m.addConstrs((v.E_BES_T[t] <= cap for t in d.T), "MAX_BES_E")
        m.addConstrs(
            (v.E_BES_T[t] == p.k_BES_ini_ * cap for t in [min(d.T), max(d.T)]),
            "INI_BES",
        )
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

        sc.balances.P_EL_source_T["BES"] = lambda t: v.P_BES_out_T[t]
        sc.balances.P_EL_sink_T["BES"] = lambda t: v.P_BES_in_T[t]


@dataclass
class PV(Component):
    """Photovoltaic System"""

    P_CAPx: float = 100
    A_avail_: float = 100
    allow_new: bool = False

    def param_func(self, sc: Scenario):
        sc.param("P_PV_CAPx_", data=self.P_CAPx, doc="Existing capacity", unit="kW_peak")
        sc.prep.P_PV_profile_T(use_coords=True)
        sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")
        sc.var("P_PV_OC_T", doc="Own consumption", unit="kW_el")
        sc.param(
            "A_PV_PerPeak_",
            data=6.5,
            doc="Area efficiency of new PV",
            unit="m²/kW_peak",
            src="https://www.dachvermieten.net/wieviel-qm-dachflaeche-fuer-1-kw-kilowatt",
        )
        sc.param("A_PV_avail_", data=self.A_avail_, doc="Area available for new PV", unit="m²")
        sc.param(
            "c_PV_OC_",
            data=0.4 * 0.0688,
            doc="Renewable Energy Law (EEG) levy on own consumption",
            unit="€/kWh_el",
        )

        if sc.consider_invest:
            sc.param("z_PV_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_PV_inv_())
            sc.param(from_db=db.k_PV_RMI_)
            sc.var("P_PV_CAPn_", doc="New capacity", unit="kW_peak", ub=1e20 * sc.params.z_PV_)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        cap = p.P_PV_CAPx_ + v.P_PV_CAPn_ if sc.consider_invest else p.P_PV_CAPx_
        m.addConstrs(
            (cap * p.P_PV_profile_T[t] == v.P_PV_FI_T[t] + v.P_PV_OC_T[t] for t in d.T),
            "PV1",
        )
        if sc.consider_invest:
            m.addConstr(v.P_PV_CAPn_ <= p.A_PV_avail_ / p.A_PV_PerPeak_, "LIM_P_PV_CAPn_")
        sc.balances.P_EL_source_T["PV"] = lambda t: v.P_PV_OC_T[t]
        sc.balances.P_EG_sell_T["PV"] = lambda t: v.P_PV_FI_T[t]
        sc.balances.C_TOT_op_["PV_OC"] = (
            p.k__comp_ * p.k__dT_ * p.c_PV_OC_ / 1e3 * v.P_PV_OC_T.sum()
        )


@dataclass
class HP(Component):
    """Electric heat pump"""

    dQ_CAPx: float = 5000
    allow_new: bool = False

    def param_func(self, sc: Scenario):
        p = sc.params
        d = sc.dims

        sc.dim("C", ["30", "65"], doc="Condensing temperature levels")

        sc.param("T__amb_", data=273 + 25)
        sc.param(
            "T_HP_C_C",
            data=pd.Series([p.T__amb_ + 5, 273 + 65], d.C),
            doc="Condensation side temp.",
            unit="K",
        )
        sc.param("T_HP_E_N", data=p.T_cDem_in_N - 5, doc="Evaporation side temp.", unit="K")
        sc.param("eta_HP_", data=0.5, doc="Ratio of reaching the ideal COP (exergy efficiency)")
        sc.param(
            "cop_HP_carnot_CN",
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
                    else p.cop_HP_carnot_CN[c, n] * p.eta_HP_
                    for c in d.C
                    for n in d.N
                }
            ),
            doc="Real heating COP",
        )
        sc.param("dQ_HP_CAPx_", data=self.dQ_CAPx, doc="Existing heating capacity", unit="kW_th")
        sc.param(
            "dQ_HP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_th"
        )
        sc.var("P_HP_TCN", doc="Consuming power", unit="kW_el")
        sc.var("dQ_HP_Cond_TCN", doc="Heat flow released on condensation side", unit="kW_th")
        sc.var("dQ_HP_Eva_TCN", doc="Heat flow absorbed on evaporation side", unit="kW_th")
        sc.var("Y_HP_TCN", doc="If source and sink are connected at time-step", vtype=GRB.BINARY)

        if sc.consider_invest:
            sc.param("z_HP_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.k_HP_RMI_)
            sc.param(from_db=db.funcs.c_HP_inv_())
            sc.var("dQ_HP_CAPn_", doc="New heating capacity", unit="kW_th", ub=1e6 * p.z_HP_)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        m.addConstrs(
            (
                v.dQ_HP_Cond_TCN[t, c, n] == v.P_HP_TCN[t, c, n] * p.cop_HP_CN[c, n]
                for t in d.T
                for c in d.C
                for n in d.N
            ),
            "HP_BAL_1",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_TCN[t, c, n] == v.dQ_HP_Eva_TCN[t, c, n] + v.P_HP_TCN[t, c, n]
                for t in d.T
                for c in d.C
                for n in d.N
            ),
            "HP_BAL_2",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_TCN[t, c, n] <= v.Y_HP_TCN[t, c, n] * p.dQ_HP_max_
                for t in d.T
                for c in d.C
                for n in d.N
            ),
            "HP_BIGM",
        )
        cap = p.dQ_HP_CAPx_ + v.dQ_HP_CAPn_ if sc.consider_invest else p.dQ_HP_CAPx_
        m.addConstrs(
            (v.dQ_HP_Cond_TCN[t, c, n] <= cap for t in d.T for c in d.C for n in d.N),
            "HP_CAP",
        )
        m.addConstrs((v.Y_HP_TCN.sum(t, "*", "*") <= 1 for t in d.T), "HP_maxOneOperatingMode")

        sc.balances.P_EL_sink_T["HP"] = lambda t: v.P_HP_TCN.sum(t, "*", "*")


@dataclass
class P2H(Component):
    """Power to heat"""

    dQ_CAPx: float = 10000

    def param_func(self, sc: Scenario):
        sc.param("dQ_P2H_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param(from_db=db.eta_P2H_)
        sc.var("P_P2H_T", doc="Consuming power", unit="kW_el")
        sc.var("dQ_P2H_T", doc="Producing heat flow", unit="kW_th")
        if sc.consider_invest:
            sc.param("z_P2H_", data=0, doc="If new capacity is allowed")
            sc.param(from_db=db.c_P2H_inv_)
            sc.param("k_P2H_RMI_", data=0)
            sc.var("dQ_P2H_CAPn_", doc="New capacity", unit="kW_th", ub=1e6 * sc.params.z_P2H_)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        cap = p.dQ_P2H_CAPx_ + v.dQ_P2H_CAPn_ if sc.consider_invest else p.dQ_P2H_CAPx_
        m.addConstrs((v.dQ_P2H_T[t] == p.eta_P2H_ * v.P_P2H_T[t] for t in d.T), "BAL_P2H")
        m.addConstrs((v.dQ_P2H_T[t] <= cap for t in d.T), "CAP_P2H")

        sc.balances.dQ_source_TL["P2H"] = lambda t, l: v.dQ_P2H_T[t] if l == "90/70" else 0
        sc.balances.P_EL_sink_T["P2H"] = lambda t: v.P_P2H_T[t]


@dataclass
class CHP(Component):
    """Combined heat and power"""

    P_CAPx: float = 0

    def param_func(self, sc: Scenario):
        sc.param("P_CHP_CAPx_", data=self.P_CAPx, doc="Existing capacity", unit="kW_el")
        sc.param(
            "P_CHP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_el"
        )
        sc.param("z_CHP_minPL_", data=1, doc="If minimal part load is modeled.")
        sc.param(from_db=db.funcs.eta_CHP_el_(fuel="ng"))
        sc.param(from_db=db.funcs.eta_CHP_th_(fuel="ng"))
        sc.param(
            "c_CHP_OC_",
            data=0.4 * 0.0688,
            doc="Renewable Energy Law (EEG) levy on own consumption",
            unit="€/kWh_el",
        )
        sc.var("dQ_CHP_T", doc="Producing heat flow", unit="kW_th")
        sc.var("F_CHP_TF", doc="Consumed fuel flow", unit="kW")
        sc.var("P_CHP_FI_T", doc="Feed-in", unit="kW_el")
        sc.var("P_CHP_OC_T", doc="Own consumption", unit="kW_el")
        sc.var("P_CHP_T", doc="Producing power", unit="kW_el")

        if sc.params.z_CHP_minPL_:
            sc.param("k_CHP_minPL_", data=0.5, doc="Minimal allowed part load")
            sc.var("Y_CHP_T", doc="If in operation", vtype=GRB.BINARY)

        if sc.consider_invest:
            sc.param("z_CHP_", data=0, doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
            sc.param(from_db=db.k_CHP_RMI_)
            sc.var("P_CHP_CAPn_", doc="New capacity", unit="kW_el", ub=1e6 * sc.params.z_CHP_)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        m.addConstrs(
            (v.P_CHP_T[t] == p.eta_CHP_el_ * quicksum(v.F_CHP_TF[t, f] for f in d.F) for t in d.T),
            "CHP_E",
        )
        m.addConstrs(
            (v.dQ_CHP_T[t] == p.eta_CHP_th_ * quicksum(v.F_CHP_TF[t, f] for f in d.F) for t in d.T),
            "CHP_Q",
        )
        cap = p.P_CHP_CAPx_ + v.P_CHP_CAPn_ if sc.consider_invest else p.P_CHP_CAPx_
        m.addConstrs((v.P_CHP_T[t] <= cap for t in d.T), "CHP_CAP")
        m.addConstrs((v.P_CHP_T[t] == v.P_CHP_FI_T[t] + v.P_CHP_OC_T[t] for t in d.T), "CHP_OC_FI")

        if p.z_CHP_minPL_:
            m.addConstrs((v.P_CHP_T[t] <= v.Y_CHP_T[t] * p.P_CHP_max_ for t in d.T), "DEF_Y_CHP_T")
            m.addConstrs(
                (
                    v.P_CHP_T[t] >= p.k_CHP_minPL_ * cap - p.P_CHP_max_ * (1 - v.Y_CHP_T[t])
                    for t in d.T
                ),
                "DEF_CHP_minPL",
            )

        sc.balances.P_EL_source_T["CHP"] = lambda t: v.P_CHP_T[t]
        sc.balances.dQ_source_TL["CHP"] = lambda t, l: v.dQ_CHP_T[t] if l == "90/70" else 0
        sc.balances.P_EG_sell_T["CHP"] = lambda t: v.P_CHP_FI_T[t]
        sc.balances.F_fuel_F["CHP"] = lambda f: v.F_CHP_TF.sum("*", f)
        sc.balances.C_TOT_op_["CHP_OC"] = (
            p.k__comp_ * p.k__dT_ * p.c_CHP_OC_ / 1e3 * v.P_CHP_OC_T.sum()
        )


@dataclass
class HOB(Component):
    """Heat-only boiler"""

    dQ_CAPx: float = 5000
    eta: float = 0.9

    def param_func(self, sc: Scenario):
        sc.param("dQ_HOB_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param("eta_HOB_", data=self.eta, doc="Thermal efficiency", unit="kWh_th/kWh")
        sc.var("dQ_HOB_T", doc="Ouput heat flow", unit="kW_th")
        sc.var("F_HOB_TF", doc="Input fuel flow", unit="kW")

        if sc.consider_invest:
            sc.param(from_db=db.funcs.c_HOB_inv_())
            sc.param(from_db=db.k_HOB_RMI_)
            sc.var("dQ_HOB_CAPn_", doc="New capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        cap = v.dQ_HOB_CAPn_ + p.dQ_HOB_CAPx_ if sc.consider_invest else p.dQ_HOB_CAPx_
        m.addConstrs((v.dQ_HOB_T[t] == v.F_HOB_TF.sum(t, "*") * p.eta_HOB_ for t in d.T), "BAL_HOB")
        m.addConstrs((v.dQ_HOB_T[t] <= cap for t in d.T), "CAP_HOB")

        sc.balances.dQ_source_TL["HOB"] = lambda t, l: v.dQ_HOB_T[t] if l == "90/70" else 0
        sc.balances.F_fuel_F["HOB"] = lambda f: v.F_HOB_TF.sum("*", f)


@dataclass
class TES(Component):
    """Thermal energy storage"""

    def param_func(self, sc: Scenario):
        d = sc.dims
        if sc.has_thermal_entities:
            L = []
            if hasattr(d, "N"):
                L += d.N
            if hasattr(d, "H"):
                L += d.H
            sc.dim("L", L, doc="Thermal demand temperature levels (inlet / outlet) in °C")

        sc.param("Q_TES_CAPx_L", fill=0, doc="Existing capacity", unit="kWh_th")
        sc.param("eta_TES_time_", data=0.995, doc="Storing efficiency")
        sc.param("k_TES_inPerCap_", data=0.5, doc="Ratio loading power / capacity")
        sc.param("k_TES_outPerCap_", data=0.5, doc="Ratio loading power / capacity")
        sc.param("k_TES_ini_L", fill=0, doc="Initial and final energy level share")
        sc.var("dQ_TES_in_TL", doc="Storage input heat flow", unit="kW_th", lb=-GRB.INFINITY)
        sc.var("Q_TES_TL", doc="Stored heat", unit="kWh_th")

        if sc.consider_invest:
            sc.param("z_TES_L", fill=0, doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_TES_inv_(estimated_size=100, temp_spread=40))
            sc.param(from_db=db.k_TES_RMI_)
            sc.var("Q_TES_CAPn_L", doc="New capacity", unit="kWh_th", ub=1e7 * sc.params.z_TES_L)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        cap = (
            lambda l: p.Q_TES_CAPx_L[l] + v.Q_TES_CAPn_L[l]
            if sc.consider_invest
            else p.Q_TES_CAPx_L[l]
        )
        m.addConstrs(
            (
                v.Q_TES_TL[t, l]
                == p.eta_TES_time_ * v.Q_TES_TL[t - 1, l] + p.k__dT_ * v.dQ_TES_in_TL[t - 1, l]
                for t in d.T[1:]
                for l in d.L
            ),
            "TES1",
        )
        m.addConstrs(
            (v.Q_TES_TL[t, l] <= cap(l) for t in d.T for l in d.L),
            "TES3",
        )
        m.addConstrs(
            (v.dQ_TES_in_TL[t, l] <= p.k_TES_inPerCap_ * cap(l) for t in d.T for l in d.L),
            "MAX_TES_IN",
        )
        m.addConstrs(
            (v.dQ_TES_in_TL[t, l] >= -p.k_TES_outPerCap_ * cap(l) for t in d.T for l in d.L),
            "MAX_TES_OUT",
        )
        m.addConstrs(
            (
                v.Q_TES_TL[t, l] == p.k_TES_ini_L[l] * cap(l)
                for t in [min(d.T), max(d.T)]
                for l in d.L
            ),
            "INI_TES",
        )

        sc.balances.dQ_sink_TL["TES"] = lambda t, l: v.dQ_TES_in_TL[t, l] if l == "90/70" else 0


@dataclass
class H2H1(Component):
    """Heat downgrading from H2 to H1"""

    def param_func(self, sc: Scenario):
        sc.var("dQ_H2H1_T", doc="Heat down-grading", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        sc.balances.dQ_sink_TL["H2H1"] = lambda t, l: v.dQ_H2H1_T[t] if l == "90/70" else 0
        sc.balances.dQ_source_TL["H2H1"] = lambda t, l: v.dQ_H2H1_T[t] if l == "60/40" else 0


@dataclass
class BEV(Component):
    """Battery electric Vehicle"""

    E_CAPx: float = 100
    z_V2X: bool = False
    z_smart: bool = False

    def param_func(self, sc: Scenario):
        p = sc.params
        d = sc.dims
        sc.dim("B", data=[1, 2], doc="BEV batteries")
        sc.param("E_BEV_Cap1Bat_B", fill=100, doc="Capacity of one battery", unit="kWh_el")
        sc.param("n_BEV_nBats_B", fill=10, doc="Number of batteries")
        sc.param(
            "E_BEV_CAPx_B",
            fill=p.E_BEV_Cap1Bat_B * p.n_BEV_nBats_B,
            doc="Capacity of all batteries",
            unit="kWh_el",
        )
        sc.param(
            "eta_BEV_time_",
            data=1.0,
            doc="Storing efficiency. Must be 1.0 for the uncontrolled charging in REF",
        )
        sc.param("eta_BEV_cycle_", data=0.9, doc="Cycle efficiency")
        sc.param("P_BEV_drive_TB", fill=0, doc="Power use", unit="kW_el")
        sc.param(
            "y_BEV_avail_TB",
            fill=1,
            doc="If BEV is available for charging at time step",
        )
        sc.param("k_BEV_inPerCap_B", fill=0.4, doc="Maximum charging power per capacity")
        sc.param("k_BEV_v2xPerCap_B", fill=0.4, doc="Maximum v2x discharging power per capacity")
        sc.param("k_BEV_empty_B", fill=0.05, doc="Definition of empty")
        sc.param("k_BEV_full_B", fill=0.95, doc="Definition of full")
        sc.param("z_BEV_smart_", data=int(self.z_smart), doc="If smart charging is allowed")
        sc.param("z_BEV_V2X_", data=int(self.z_V2X), doc="If vehicle-to-X is allowed")
        sc.param("k_BEV_ini_B", fill=0.7, doc="Initial and final state of charge")
        sc.var("E_BEV_TB", doc="Electricity stored in BEV battery", unit="kWh_el")
        sc.var("P_BEV_in_TB", doc="Charging power", unit="kW_el")
        sc.var("P_BEV_V2X_TB", doc="Discharging power for vehicle-to-X", unit="kW_el")
        sc.var("x_BEV_penalty_", doc="Penalty to ensure uncontrolled charging in REF")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars):
        T = d.T
        B = d.B

        if sc.params.P_BEV_drive_TB.sum() == 0:
            logger.warning("P_BEV_drive_TB is all zero. Please set data")
        if sc.params.y_BEV_avail_TB.sum() == 1:
            logger.warning("y_BEV_avail_TB is all one. Please set data")

        m.addConstrs(
            (
                v.E_BEV_TB[t, b]
                == v.E_BEV_TB[t - 1, b] * p.eta_BEV_time_
                + p.k__dT_
                * (
                    v.P_BEV_in_TB[t, b] * p.eta_BEV_cycle_
                    - p.P_BEV_drive_TB[t, b]
                    - v.P_BEV_V2X_TB[t, b]
                )
                for t in T[1:]
                for b in B
            ),
            "BAL_BEV",
        )
        m.addConstrs(
            (v.E_BEV_TB[t, b] <= p.k_BEV_full_B[b] * p.E_BEV_CAPx_B[b] for t in T for b in B),
            "MAX_E_BEV",
        )
        m.addConstrs(
            (v.E_BEV_TB[t, b] >= p.k_BEV_empty_B[b] * p.E_BEV_CAPx_B[b] for t in T for b in B),
            "MIN_E_BEV",
        )

        m.addConstrs(
            (
                v.E_BEV_TB[t, b] == p.k_BEV_ini_B[b] * p.E_BEV_CAPx_B[b]
                for t in [min(T), max(T)]
                for b in B
            ),
            "INI_BEV",
        )
        m.addConstr(
            (
                v.x_BEV_penalty_
                == (1 - p.z_BEV_smart_) * quicksum(t * v.P_BEV_in_TB[t, b] for t in T for b in B)
            ),
            "PENALTY_BEV",
        )
        m.addConstrs(
            (
                v.P_BEV_in_TB[t, b]
                <= p.y_BEV_avail_TB[t, b] * p.E_BEV_CAPx_B[b] * p.k_BEV_inPerCap_B[b]
                for t in T
                for b in B
            ),
            "MAX_BEV_in",
        )
        m.addConstrs(
            (
                v.P_BEV_V2X_TB[t, b]
                <= p.y_BEV_avail_TB[t, b]
                * p.z_BEV_V2X_
                * p.E_BEV_CAPx_B[b]
                * p.k_BEV_v2xPerCap_B[b]
                for t in T
                for b in B
            ),
            "MAX_BEV_v2x",
        )

        sc.balances.P_EL_source_T["BEV"] = lambda t: v.P_BEV_V2X_TB.sum(t, "*")
        sc.balances.P_EL_sink_T["BEV"] = lambda t: v.P_BEV_in_TB.sum(t, "*")
        sc.balances.Penalty_["BEV"] = v.x_BEV_penalty_


dependencies = [
    ("PRE", {}),
    ("cDem", {"PRE"}),
    ("hDem", {"PRE"}),
    ("eDem", {"PRE"}),
    ("EG", {"PRE", "PV", "CHP"}),
    ("Fuel", {"PRE"}),
    ("BES", {"PRE"}),
    ("BEV", {"PRE"}),
    ("PV", {"PRE"}),
    ("P2H", {"PRE"}),
    ("CHP", {"PRE"}),
    ("HOB", {"PRE"}),
    ("H2H1", {"PRE"}),
    ("HP", {"PRE", "cDem", "hDem"}),
    ("TES", {"PRE", "cDem", "hDem"}),
]
dependencies.append(("POST", [x[0] for x in dependencies]))

set_component_order_by_dependency(dependencies=dependencies, classes=globals())
