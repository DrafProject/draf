import datetime
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from gurobipy import GRB, Model, quicksum

from draf import Collectors, Dimensions, Params, Results, Scenario, Vars

# from draf.model_builder import autocollectors
from draf.abstract_component import Component
from draf.conventions import Descs
from draf.helper import conv, get_annuity_factor, set_component_order_by_order_restrictions
from draf.prep import DataBase as db

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


@dataclass
class Main(Component):
    """Objective functions and general collectors. This must be the last model_func to be executed."""

    def param_func(self, sc: Scenario):
        sc.collector("P_EL_source_T", doc="Power sources", unit="kW_el")
        sc.collector("P_EL_sink_T", doc="Power sinks", unit="kW_el")
        sc.collector("dQ_cooling_source_TN", doc="Cooling energy flow sources", unit="kW_th")
        sc.collector("dQ_cooling_sink_TN", doc="Cooling energy flow sinks", unit="kW_th")
        sc.collector("dQ_heating_source_TH", doc="Heating energy flow sources", unit="kW_th")
        sc.collector("dQ_heating_sink_TH", doc="Heating energy flow sinks", unit="kW_th")
        sc.collector("C_TOT_", doc="Total costs", unit="k€/a")
        sc.collector("C_TOT_op_", doc="Total operating costs", unit="k€/a")
        sc.collector("CE_TOT_", doc="Total carbon emissions", unit="kgCO2eq/a")
        sc.collector("Penalty_", doc="Penalty term for objective function", unit="Any")

        if sc.consider_invest:
            sc.collector("C_TOT_RMI_", doc="Total annual maintainance cost", unit="k€/a")
            sc.collector("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.collector("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")

        sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

        if sc.consider_invest:
            sc.param("k__r_", data=0.06, doc="Calculatory interest rate")
            sc.var("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.var("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")
            sc.var("C_TOT_RMI_", doc="Total annual maintainance cost", unit="k€")
        sc.var("Penalty_", "Penalty term for objective function")

        sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
        sc.param("k_PTO_C_", data=1, doc="Normalization factor")
        sc.param("k_PTO_CE_", data=1 / 1e4, doc="Normalization factor")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.setObjective(
            (
                (1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_
                + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_
                + v.Penalty_
            ),
            GRB.MINIMIZE,
        )

        m.addConstr(v.C_TOT_op_ == quicksum(c.C_TOT_op_.values()), "operating_cost_balance")
        c.C_TOT_["op"] = v.C_TOT_op_

        if sc.consider_invest:
            m.addConstr(v.C_TOT_inv_ == quicksum(c.C_TOT_inv_.values()), "investment_cost")
            m.addConstr(v.C_TOT_RMI_ == quicksum(c.C_TOT_RMI_.values()), "repair_cost")
            m.addConstr(
                v.C_TOT_invAnn_ == quicksum(c.C_TOT_invAnn_.values()), "annualized_investment_cost"
            )
            c.C_TOT_["RMI"] = v.C_TOT_RMI_
            c.C_TOT_["inv"] = v.C_TOT_invAnn_

            ## AUTOCOLLECTORS (currently unused) ---------------------------------------------------
            # m.addConstr( (v.C_TOT_inv_ == autocollectors.C_inv_(p, v, r=p.k__r_) * conv("€", "k€", 1e-3)) )
            # m.addConstr( (v.C_TOT_invAnn_ == autocollectors.C_inv_Annual_(p, v, r=p.k__r_) * conv("€", "k€", 1e-3)) )
            # m.addConstr( (v.C_TOT_RMI_ == autocollectors.C_TOT_RMI_(p, v) * conv("€", "k€", 1e-3)), "DEF_C_TOT_RMI_", )
            ## -------------------------------------------------------------------------------------

        m.addConstr(v.C_TOT_ == quicksum(c.C_TOT_.values()), "total_cost_balance")
        m.addConstr(
            v.CE_TOT_ == p.k__PartYearComp_ * quicksum(c.CE_TOT_.values()),
            "carbon_emission_balance",
        )
        m.addConstr(v.Penalty_ == quicksum(c.Penalty_.values()), "penalty_balance")
        m.addConstrs(
            (
                quicksum(x(t) for x in c.P_EL_source_T.values())
                == quicksum(x(t) for x in c.P_EL_sink_T.values())
                for t in d.T
            ),
            "electricity_balance",
        )

        if hasattr(d, "N"):
            m.addConstrs(
                (
                    quicksum(x(t, n) for x in c.dQ_cooling_source_TN.values())
                    == quicksum(x(t, n) for x in c.dQ_cooling_sink_TN.values())
                    for t in d.T
                    for n in d.N
                ),
                "cool_balance",
            )

        if hasattr(d, "H"):
            m.addConstrs(
                (
                    quicksum(x(t, h) for x in c.dQ_heating_source_TH.values())
                    == quicksum(x(t, h) for x in c.dQ_heating_sink_TH.values())
                    for t in d.T
                    for h in d.H
                ),
                "heat_balance",
            )


@dataclass
class cDem(Component):
    """Cooling demand"""

    def param_func(self, sc: Scenario):
        sc.dim("N", data=[1, 2], doc="Cooling temperature levels (inlet / outlet) in °C")

        sc.param(name="dQ_cDem_TN", fill=0, doc="Cooling demand", unit="kW_th")
        sc.params.dQ_cDem_TN.loc[:, 1] = sc.prep.dQ_cDem_T(annual_energy=1e4).values
        sc.param("T_cDem_in_N", data=[7, 30], doc="Cooling inlet temperature", unit="°C")
        sc.param("T_cDem_out_N", data=[12, 35], doc="Cooling outlet temperature", unit="°C")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_cooling_source_TN["cDem"] = lambda t, n: p.dQ_cDem_TN[t, n]


@dataclass
class hDem(Component):
    """Heating demand"""

    def param_func(self, sc: Scenario):
        sc.dim("H", [1, 2], doc="Heating temperature levels (inlet / outlet) in °C")

        sc.param(name="dQ_hDem_TH", fill=0, doc="Heating demand", unit="kW_th")
        sc.params.dQ_hDem_TH.loc[:, 1] = sc.prep.dQ_hDem_T(annual_energy=1e6).values
        sc.param("T_hDem_in_H", data=[60, 90], doc="Heating inlet temperature", unit="°C")
        sc.param("T_hDem_out_H", data=[40, 70], doc="Heating outlet temperature", unit="°C")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_heating_sink_TH["hDem"] = lambda t, h: p.dQ_hDem_TH[t, h]


@dataclass
class eDem(Component):
    """Electricity demand"""

    p_el: Optional[pd.Series] = None
    profile: str = "G3"
    annual_energy: float = 5e6

    def param_func(self, sc: Scenario):
        if self.p_el is None:
            sc.prep.P_eDem_T(profile=self.profile, annual_energy=self.annual_energy)
        else:
            sc.param("P_eDem_T", data=self.p_el, doc="Electricity demand", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.P_EL_sink_T["eDem"] = lambda t: p.P_eDem_T[t]


@dataclass
class EG(Component):
    """Electricity grid"""

    c_buyPeak: float = 50.0
    prepared_tariffs: Tuple = ("FLAT", "TOU", "RTP")
    selected_tariff: str = "RTP"
    consider_intensiveGridUse: bool = False

    def param_func(self, sc: Scenario):
        sc.collector("P_EG_sell_T", doc="Sold electricity power", unit="kW_el")
        sc.param("c_EG_buyPeak_", data=self.c_buyPeak, doc="Peak price", unit="€/kW_el/a")

        if "RTP" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_T()
        if "TOU" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_T()
            sc.prep.c_EG_TOU_T()
        if "FLAT" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_T()
            sc.prep.c_EG_FLAT_T()
        sc.param(
            "c_EG_T",
            data=getattr(sc.params, f"c_EG_{self.selected_tariff}_T"),
            doc="Chosen electricity tariff",
            unit="€/kWh_el",
        )
        sc.prep.c_EG_addon_()
        sc.prep.ce_EG_T()
        sc.param("t_EG_minFLH_", data=0, doc="Minimal full load hours", unit="h")
        sc.var("P_EG_buy_T", doc="Purchased electrical power", unit="kW_el")
        sc.var("P_EG_sell_T", doc="Selling electrical power", unit="kW_el")
        sc.var("P_EG_buyPeak_", doc="Peak electrical power", unit="kW_el")

        if self.consider_intensiveGridUse:
            # FIXME: Currently not working
            sc.dim(
                "G",
                data=["7000-7500", "7500-8000", ">8000"],
                doc="Full load hour sections for indensive grid use",
            )
            sc.var("y_EG_FLH_G", doc="If full load hour section applies", vtype=GRB.BINARY)
            sc.param("t_EG_minFLH_G", data=[7000, 7500, 8000], unit="h")
            sc.param(
                "k_EG_FLH_G",
                data=[0.8, 0.85, 0.9],
                doc="Peak price reduction factor if full load hour section applies",
                src="@Tieman_2020",
            )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (v.P_EG_sell_T[t] == sum(x(t) for x in c.P_EG_sell_T.values()) for t in d.T), "EG_sell"
        )
        m.addConstrs((v.P_EG_buy_T[t] <= v.P_EG_buyPeak_ for t in d.T), "EG_peak_price")

        if p.t_EG_minFLH_ > 0:
            m.addConstr(
                v.P_EG_buy_T.sum() * p.k__dT_ * p.k__PartYearComp_
                >= p.t_EG_minFLH_ * v.P_EG_buyPeak_,
                "EG_minimum_full_load_hours",
            )

        if self.consider_intensiveGridUse:
            m.addConstrs(
                (
                    v.P_EG_buy_T.sum() * p.k__dT_ * p.k__PartYearComp_
                    >= p.t_EG_minFLH_G[g] * v.y_EG_FLH_G[g] * v.P_EG_buyPeak_
                    for g in d.G
                ),
                "EG_intensive_grid_use",
            )
            m.addConstr(v.y_EG_FLH_G.sum() <= 1, "DEF_FLH_2")

        c.P_EL_source_T["EG"] = lambda t: v.P_EG_buy_T[t]
        c.P_EL_sink_T["EG"] = lambda t: v.P_EG_sell_T[t]
        igu_factor = (1 - v.y_EG_FLH_G.prod(p.k_EG_FLH_G)) if self.consider_intensiveGridUse else 1
        c.C_TOT_op_["EG_peak"] = (
            v.P_EG_buyPeak_ * igu_factor * p.c_EG_buyPeak_ * conv("€", "k€", 1e-3)
        )
        c.C_TOT_op_["EG"] = (
            p.k__dT_
            * p.k__PartYearComp_
            * quicksum(
                v.P_EG_buy_T[t] * (p.c_EG_T[t] + p.c_EG_addon_) - v.P_EG_sell_T[t] * p.c_EG_T[t]
                for t in d.T
            )
            * conv("€", "k€", 1e-3)
        )
        c.CE_TOT_["EG"] = p.k__dT_ * quicksum(
            p.ce_EG_T[t] * (v.P_EG_buy_T[t] - v.P_EG_sell_T[t]) for t in d.T
        )

    def postprocess_func(self, r: Results):
        r.make_pos_ent("P_EG_buy_T")


@dataclass
class Fuel(Component):
    """Fuels"""

    c_ceTax: float = 55

    def param_func(self, sc: Scenario):
        sc.dim("F", ["ng", "bio"], doc="Types of fuel")
        sc.collector("F_fuel_F", doc="Fuel power", unit="kWh")
        sc.param(from_db=db.c_Fuel_F)
        sc.param("c_Fuel_ceTax_", data=self.c_ceTax, doc="Carbon tax on fuel", unit="€/tCO2eq")
        sc.param(from_db=db.ce_Fuel_F)
        sc.var("C_Fuel_ceTax_", doc="Total carbon tax on fuel", unit="k€/a")
        sc.var("CE_Fuel_", doc="Total carbon emissions for fuel", unit="kgCO2eq/a")
        sc.var("C_Fuel_", doc="Total cost for fuel", unit="k€/a")
        sc.var("F_fuel_F", doc="Total fuel consumption", unit="kW")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (v.F_fuel_F[f] == quicksum(x(f) for x in c.F_fuel_F.values()) for f in d.F),
            "fuel_balance",
        )
        m.addConstr(v.CE_Fuel_ == p.k__dT_ * v.F_fuel_F.prod(p.ce_Fuel_F))
        m.addConstr(v.C_Fuel_ == p.k__dT_ * v.F_fuel_F.prod(p.c_Fuel_F) * conv("€", "k€", 1e-3))
        m.addConstr(
            v.C_Fuel_ceTax_
            == p.c_Fuel_ceTax_ * conv("/t", "(/kg", 1e-3) * v.CE_Fuel_ * conv("€", "k€", 1e-3)
        )
        c.CE_TOT_["Fuel"] = v.CE_Fuel_
        c.C_TOT_op_["Fuel"] = p.k__PartYearComp_ * v.C_Fuel_
        c.C_TOT_op_["FuelCeTax"] = p.k__PartYearComp_ * v.C_Fuel_ceTax_


@dataclass
class BES(Component):
    """Battery Energy Storage"""

    E_CAPx: float = 0
    allow_new: bool = True

    def param_func(self, sc: Scenario):
        sc.param("E_BES_CAPx_", data=self.E_CAPx, doc="Existing capacity", unit="kWh_el")
        sc.param("k_BES_ini_", data=0, doc="Initial and final energy filling share")
        sc.param(
            "eta_BES_ch_",
            data=db.eta_BES_cycle_.data ** 0.5,
            doc="Charging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(
            "eta_BES_dis_",
            data=db.eta_BES_cycle_.data ** 0.5,
            doc="Discharging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(from_db=db.eta_BES_time_)
        sc.param(from_db=db.k_BES_inPerCap_)
        sc.param(from_db=db.k_BES_outPerCap_)
        sc.var("E_BES_T", doc="Electricity stored", unit="kWh_el")
        sc.var("P_BES_in_T", doc="Charging power", unit="kW_el")
        sc.var("P_BES_out_T", doc="Discharging power", unit="kW_el")

        if sc.consider_invest:
            sc.param(from_db=db.k_BES_RMI_)
            sc.param(from_db=db.N_BES_)
            sc.param("z_BES_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_BES_inv_(estimated_size=100, which="mean"))
            sc.var("E_BES_CAPn_", doc="New capacity", unit="kWh_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        """Note: In this model does not prevent simultaneous charging and discharging,
        which can appear negativ electricity prices. To avoid this behaviour expensive binary
        variables can be introduced, e.g., like in
        AmirMansouri.2021: https://doi.org/10.1016/j.seta.2021.101376
        """

        cap = p.E_BES_CAPx_ + v.E_BES_CAPn_ if sc.consider_invest else p.E_BES_CAPx_
        m.addConstrs(
            (v.P_BES_in_T[t] <= p.k_BES_inPerCap_ * cap for t in d.T), "BES_limit_charging_power"
        )
        m.addConstrs(
            (v.P_BES_out_T[t] <= p.k_BES_outPerCap_ * cap for t in d.T),
            "BES_limit_discharging_power",
        )
        m.addConstrs((v.E_BES_T[t] <= cap for t in d.T), "BES_limit_cap")
        m.addConstr((v.E_BES_T[d.T[-1]] == p.k_BES_ini_ * cap), "BES_last_timestep")
        m.addConstrs(
            (
                v.E_BES_T[t]
                == (p.k_BES_ini_ * cap if t == d.T[0] else v.E_BES_T[t - 1]) * p.eta_BES_time_
                + (v.P_BES_in_T[t] * p.eta_BES_ch_ - v.P_BES_out_T[t] / p.eta_BES_dis_) * p.k__dT_
                for t in d.T
            ),
            "BES_electricity_balance",
        )
        c.P_EL_source_T["BES"] = lambda t: v.P_BES_out_T[t]
        c.P_EL_sink_T["BES"] = lambda t: v.P_BES_in_T[t]
        if sc.consider_invest:
            m.addConstr((v.E_BES_CAPn_ <= p.z_BES_ * 1e6), "BES_limit_new_capa")
            C_inv_ = v.E_BES_CAPn_ * p.c_BES_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["BES"] = C_inv_
            c.C_TOT_invAnn_["BES"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_BES_)
            c.C_TOT_RMI_["BES"] = C_inv_ * p.k_BES_RMI_


@dataclass
class PV(Component):
    """Photovoltaic System"""

    P_CAPx: float = 0
    A_avail_: float = 100
    allow_new: bool = True

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
            src="@BMWI_2020",
        )

        if sc.consider_invest:
            sc.param("z_PV_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_PV_inv_())
            sc.param(from_db=db.k_PV_RMI_)
            sc.param(from_db=db.N_PV_)
            sc.var("P_PV_CAPn_", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.P_PV_CAPx_ + v.P_PV_CAPn_ if sc.consider_invest else p.P_PV_CAPx_
        m.addConstrs(
            (cap * p.P_PV_profile_T[t] == v.P_PV_FI_T[t] + v.P_PV_OC_T[t] for t in d.T),
            "PV_balance",
        )
        c.P_EL_source_T["PV"] = lambda t: v.P_PV_OC_T[t]
        c.P_EG_sell_T["PV"] = lambda t: v.P_PV_FI_T[t]
        c.C_TOT_op_["PV_OC"] = (
            p.k__dT_ * p.k__PartYearComp_ * p.c_PV_OC_ * v.P_PV_OC_T.sum() * conv("€", "k€", 1e-3)
        )

        if sc.consider_invest:
            m.addConstr(v.P_PV_CAPn_ <= p.z_PV_ * p.A_PV_avail_ / p.A_PV_PerPeak_, "PV_limit_capn")
            C_inv_ = v.P_PV_CAPn_ * p.c_PV_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["PV"] = C_inv_
            c.C_TOT_invAnn_["PV"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_PV_)
            c.C_TOT_RMI_["PV"] = C_inv_ * p.k_PV_RMI_


@dataclass
class HP(Component):
    """Electric heat pump"""

    dQ_CAPx: float = 0
    allow_new: bool = True
    time_dependent_amb: bool = True
    n: int = 1
    heating_levels: Optional[List] = None
    cooling_levels: Optional[List] = None
    ambient_as_sink: bool = True
    ambient_as_source: bool = True

    def param_func(self, sc: Scenario):
        p = sc.params
        d = sc.dims

        def get_E():
            e = ["E_amb"] if self.ambient_as_source else []
            e += sc.dims.N if self.cooling_levels is None else self.cooling_levels
            return e

        def get_C():
            c = ["C_amb"] if self.ambient_as_sink else []
            c += sc.dims.H if self.heating_levels is None else self.heating_levels
            return c

        sc.dim("E", data=get_E(), doc="Evaporation temperature levels")
        sc.dim("C", data=get_C(), doc="Condensing temperature levels")
        sc.collector("dQ_amb_sink_", doc="Thermal energy flow to ambient", unit="kW_th")
        sc.collector("dQ_amb_source_", doc="Thermal energy flow from ambient", unit="kW_th")

        if self.time_dependent_amb:
            sc.prep.T__amb_T()
        sc.param("T__amb_", data=25, doc="Approximator for ambient air", unit="°C")

        sc.param(
            "T_HP_Cond_C", data=p.T_hDem_in_H + 5, doc="Condensation side temperature", unit="°C"
        )
        sc.param(
            "T_HP_Eva_E", data=p.T_cDem_in_N - 5, doc="Evaporation side temperature", unit="°C"
        )
        sc.param("n_HP_", data=self.n, doc="Number of existing heat pumps")
        sc.param("eta_HP_", data=0.5, doc="Ratio of reaching the ideal COP (exergy efficiency)")
        sc.param("dQ_HP_CAPx_", data=self.dQ_CAPx, doc="Existing heating capacity", unit="kW_th")
        sc.param(
            "dQ_HP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_th"
        )
        sc.var("P_HP_TEC", doc="Consuming power", unit="kW_el")
        sc.var("dQ_HP_Cond_TEC", doc="Heat flow released on condensation side", unit="kW_th")
        sc.var("dQ_HP_Eva_TEC", doc="Heat flow absorbed on evaporation side", unit="kW_th")
        sc.var("Y_HP_TEC", doc="If source and sink are connected at time-step", vtype=GRB.BINARY)

        if sc.consider_invest:
            sc.param("z_HP_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.k_HP_RMI_)
            sc.param(from_db=db.N_HP_)
            sc.param(from_db=db.funcs.c_HP_inv_())
            sc.var("dQ_HP_CAPn_", doc="New heating capacity", unit="kW_th")

    def get_cop_via_hplib(
        self, t_eva: pd.Series, t_cond: pd.Series, type: str = "air", regulated: bool = True
    ):
        """UNUSED: Get the heating COP from the hplib package https://github.com/RE-Lab-Projects/hplib

        TODO: integrate this function in model_func

        Args:
            t_eva: Evaporation temperature time series
            t_cond: Condenstaion temperature time series
            type: on of 'air', 'brine'
            regulated: If the heat pump is regulated (Otherwise On-Off)
        """

        import hplib as hpl

        group_id = 1 if type == "air" else 2
        if not regulated:
            group_id += 3
        pars = hpl.get_parameters(model="Generic", group_id=group_id, t_in=-7, t_out=52, p_th=1e4)
        results = hpl.simulate(
            t_in_primary=t_eva, t_in_secondary=t_cond, t_amb=t_eva, parameters=pars
        )
        return results["COP"]

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        def get_cop(t, e, c):
            T_amb = p.T__amb_T[t] if self.time_dependent_amb else p.T__amb_
            T_cond = T_amb + 5 if c == "C_amb" else p.T_HP_Cond_C[c]
            T_eva = T_amb - 5 if e == "E_amb" else p.T_HP_Eva_E[e]
            return 100 if T_cond <= T_eva else p.eta_HP_ * (T_cond + 273) / (T_cond - T_eva)

        m.addConstrs(
            (
                v.dQ_HP_Cond_TEC[t, e, c] == v.P_HP_TEC[t, e, c] * get_cop(t, e, c)
                for t in d.T
                for e in d.E
                for c in d.C
            ),
            "HP_balance_1",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_TEC[t, e, c] == v.dQ_HP_Eva_TEC[t, e, c] + v.P_HP_TEC[t, e, c]
                for t in d.T
                for e in d.E
                for c in d.C
            ),
            "HP_balance_2",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_TEC[t, e, c] <= v.Y_HP_TEC[t, e, c] * p.dQ_HP_max_
                for t in d.T
                for e in d.E
                for c in d.C
            ),
            "HP_bigM",
        )
        cap = p.dQ_HP_CAPx_ + v.dQ_HP_CAPn_ if sc.consider_invest else p.dQ_HP_CAPx_
        m.addConstrs((v.dQ_HP_Cond_TEC.sum(t, "*", "*") <= cap for t in d.T), "HP_limit_cap")
        m.addConstrs((v.Y_HP_TEC.sum(t, "*", "*") <= p.n_HP_ for t in d.T), "HP_operating_mode")

        c.P_EL_sink_T["HP"] = lambda t: v.P_HP_TEC.sum(t, "*", "*")
        c.dQ_cooling_sink_TN["HP"] = lambda t, n: v.dQ_HP_Eva_TEC.sum(t, n, "*")
        c.dQ_heating_source_TH["HP"] = lambda t, h: v.dQ_HP_Cond_TEC.sum(t, "*", h)
        c.dQ_amb_source_["HP"] = v.dQ_HP_Eva_TEC.sum("*", "E_amb", "*") * p.k__dT_
        c.dQ_amb_sink_["HP"] = v.dQ_HP_Eva_TEC.sum("*", "*", "C_amb") * p.k__dT_

        if sc.consider_invest:
            m.addConstr((v.dQ_HP_CAPn_ <= p.z_HP_ * 1e6), "HP_limit_capn")
            C_inv_ = v.dQ_HP_CAPn_ * p.c_HP_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["HP"] = C_inv_
            c.C_TOT_invAnn_["HP"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_HP_)
            c.C_TOT_RMI_["HP"] = C_inv_ * p.k_HP_RMI_


@dataclass
class P2H(Component):
    """Power to heat"""

    dQ_CAPx: float = 0
    allow_new: bool = True

    def param_func(self, sc: Scenario):
        sc.param("dQ_P2H_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param(from_db=db.eta_P2H_)
        sc.var("P_P2H_T", doc="Consuming power", unit="kW_el")
        sc.var("dQ_P2H_T", doc="Producing heat flow", unit="kW_th")

        if sc.consider_invest:
            sc.param(from_db=db.N_P2H_)
            sc.param("z_P2H_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.c_P2H_inv_)
            sc.param("k_P2H_RMI_", data=0, doc=Descs.RMI.en)
            sc.var("dQ_P2H_CAPn_", doc="New capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.dQ_P2H_CAPx_ + v.dQ_P2H_CAPn_ if sc.consider_invest else p.dQ_P2H_CAPx_
        m.addConstrs((v.dQ_P2H_T[t] == p.eta_P2H_ * v.P_P2H_T[t] for t in d.T), "P2H_balance")
        m.addConstrs((v.dQ_P2H_T[t] <= cap for t in d.T), "P2H_limit_heat_flow")

        c.dQ_heating_source_TH["P2H"] = lambda t, h: v.dQ_P2H_T[t] if h == 2 else 0
        c.P_EL_sink_T["P2H"] = lambda t: v.P_P2H_T[t]

        if sc.consider_invest:
            m.addConstr((v.dQ_P2H_CAPn_ <= p.z_P2H_ * 1e6), "P2H_limit_new_capa")
            C_inv_ = v.dQ_P2H_CAPn_ * p.c_P2H_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["P2H"] = C_inv_
            c.C_TOT_invAnn_["P2H"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_P2H_)
            c.C_TOT_RMI_["P2H"] = C_inv_ * p.k_P2H_RMI_


@dataclass
class CHP(Component):
    """Combined heat and power"""

    P_CAPx: float = 0
    allow_new: bool = True

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
            src="@BMWI_2020",
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
            sc.param("z_CHP_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
            sc.param(from_db=db.k_CHP_RMI_)
            sc.param(from_db=db.N_CHP_)
            sc.var("P_CHP_CAPn_", doc="New capacity", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (v.P_CHP_T[t] == p.eta_CHP_el_ * quicksum(v.F_CHP_TF[t, f] for f in d.F) for t in d.T),
            "CHP_elec_balance",
        )
        m.addConstrs(
            (v.dQ_CHP_T[t] == p.eta_CHP_th_ * quicksum(v.F_CHP_TF[t, f] for f in d.F) for t in d.T),
            "CHP_heat_balance",
        )
        cap = p.P_CHP_CAPx_ + v.P_CHP_CAPn_ if sc.consider_invest else p.P_CHP_CAPx_
        m.addConstrs((v.P_CHP_T[t] <= cap for t in d.T), "CHP_limit_elecPower")
        m.addConstrs(
            (v.P_CHP_T[t] == v.P_CHP_FI_T[t] + v.P_CHP_OC_T[t] for t in d.T),
            "CHP_feedIn_vs_ownConsumption",
        )

        if p.z_CHP_minPL_:
            m.addConstrs(
                (v.P_CHP_T[t] <= v.Y_CHP_T[t] * p.P_CHP_max_ for t in d.T),
                "CHP_minimal_part_load_1",
            )
            m.addConstrs(
                (
                    v.P_CHP_T[t] >= p.k_CHP_minPL_ * cap - p.P_CHP_max_ * (1 - v.Y_CHP_T[t])
                    for t in d.T
                ),
                "CHP_minimal_part_load_2",
            )

        c.P_EL_source_T["CHP"] = lambda t: v.P_CHP_OC_T[t]
        c.dQ_heating_source_TH["CHP"] = lambda t, h: v.dQ_CHP_T[t] if h == 2 else 0
        c.P_EG_sell_T["CHP"] = lambda t: v.P_CHP_FI_T[t]
        c.F_fuel_F["CHP"] = lambda f: v.F_CHP_TF.sum("*", f) * p.k__dT_
        c.C_TOT_op_["CHP_OC"] = (
            p.k__PartYearComp_ * p.k__dT_ * p.c_CHP_OC_ * conv("€", "k€", 1e-3) * v.P_CHP_OC_T.sum()
        )

        if sc.consider_invest:
            m.addConstr((v.P_CHP_CAPn_ <= p.z_CHP_ * 1e6), "CHP_limit_new_capa")
            C_inv_ = v.P_CHP_CAPn_ * p.c_CHP_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["CHP"] = C_inv_
            c.C_TOT_invAnn_["CHP"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_CHP_)
            c.C_TOT_RMI_["CHP"] = C_inv_ * p.k_CHP_RMI_


@dataclass
class HOB(Component):
    """Heat-only boiler"""

    dQ_CAPx: float = 0
    allow_new = True

    def param_func(self, sc: Scenario):
        sc.param("dQ_HOB_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param("eta_HOB_", from_db=db.eta_HOB_)
        sc.var("dQ_HOB_T", doc="Ouput heat flow", unit="kW_th")
        sc.var("F_HOB_TF", doc="Input fuel flow", unit="kW")

        if sc.consider_invest:
            sc.param(from_db=db.funcs.c_HOB_inv_())
            sc.param(from_db=db.k_HOB_RMI_)
            sc.param(from_db=db.N_HOB_)
            sc.param("z_HOB_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.var("dQ_HOB_CAPn_", doc="New capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.dQ_HOB_CAPx_ + v.dQ_HOB_CAPn_ if sc.consider_invest else p.dQ_HOB_CAPx_
        m.addConstrs((v.dQ_HOB_T[t] == v.F_HOB_TF.sum(t, "*") * p.eta_HOB_ for t in d.T), "HOB_bal")
        m.addConstrs((v.dQ_HOB_T[t] <= cap for t in d.T), "HOB_limit_heat_flow")

        c.dQ_heating_source_TH["HOB"] = lambda t, h: v.dQ_HOB_T[t] if h == 2 else 0
        c.F_fuel_F["HOB"] = lambda f: v.F_HOB_TF.sum("*", f) * p.k__dT_

        if sc.consider_invest:
            m.addConstr((v.dQ_HOB_CAPn_ <= p.z_HOB_ * 1e6), "HOB_limit_new_capa")
            C_inv_ = v.dQ_HOB_CAPn_ * p.c_HOB_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["HOB"] = C_inv_
            c.C_TOT_invAnn_["HOB"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_HOB_)
            c.C_TOT_RMI_["HOB"] = C_inv_ * p.k_HOB_RMI_


@dataclass
class TES(Component):
    """Thermal energy storage"""

    allow_new = True

    def param_func(self, sc: Scenario):
        d = sc.dims

        if sc.has_thermal_entities:
            L = []
            if hasattr(d, "N"):
                L += [f"N{n}" for n in d.N]
            if hasattr(d, "H"):
                L += [f"H{h}" for h in d.H]
            sc.dim("L", data=L, doc="Thermal demand temperature levels (inlet / outlet) in °C")

        sc.param("Q_TES_CAPx_L", fill=0, doc="Existing capacity", unit="kWh_th")
        sc.param("eta_TES_time_", data=0.995, doc="Storing efficiency")
        sc.param("k_TES_inPerCap_", data=0.5, doc="Ratio loading power / capacity")
        sc.param("k_TES_outPerCap_", data=0.5, doc="Ratio loading power / capacity")
        sc.param("k_TES_ini_L", fill=0.5, doc="Initial and final energy level share")
        sc.var("dQ_TES_in_TL", doc="Storage input heat flow", unit="kW_th", lb=-GRB.INFINITY)
        sc.var("Q_TES_TL", doc="Stored heat", unit="kWh_th")

        if sc.consider_invest:
            sc.param("z_TES_L", fill=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_TES_inv_(estimated_size=100, temp_spread=40))
            sc.param(from_db=db.k_TES_RMI_)
            sc.param(from_db=db.N_TES_)
            sc.var("Q_TES_CAPn_L", doc="New capacity", unit="kWh_th", ub=1e7)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = lambda l: p.Q_TES_CAPx_L[l] + (
            v.Q_TES_CAPn_L[l] if sc.consider_invest else p.Q_TES_CAPx_L[l]
        )

        m.addConstrs(
            (
                v.Q_TES_TL[t, l]
                == ((p.k_TES_ini_L[l] * cap(l)) if t == d.T[0] else v.Q_TES_TL[t - 1, l])
                * p.eta_TES_time_
                + p.k__dT_ * v.dQ_TES_in_TL[t, l]
                for t in d.T
                for l in d.L
            ),
            "TES_balance",
        )
        m.addConstrs((v.Q_TES_TL[t, l] <= cap(l) for t in d.T for l in d.L), "TES_limit_cap")
        m.addConstrs(
            (v.dQ_TES_in_TL[t, l] <= p.k_TES_inPerCap_ * cap(l) for t in d.T for l in d.L),
            "TES_limit_in",
        )
        m.addConstrs(
            (v.dQ_TES_in_TL[t, l] >= -p.k_TES_outPerCap_ * cap(l) for t in d.T for l in d.L),
            "TES_limit_out",
        )
        m.addConstrs(
            (v.Q_TES_TL[d.T[-1], l] == p.k_TES_ini_L[l] * cap(l) for l in d.L), "TES_last_timestep"
        )

        # only sink here, since dQ_TES_in_TL is also defined for negative
        # values to reduce number of variables:
        c.dQ_cooling_sink_TN["TES"] = lambda t, n: v.dQ_TES_in_TL[t, f"N{n}"]
        c.dQ_heating_sink_TH["TES"] = lambda t, h: v.dQ_TES_in_TL[t, f"H{h}"]

        if sc.consider_invest:
            m.addConstrs((v.Q_TES_CAPn_L[l] <= p.z_TES_L[l] * 1e5 for l in d.L), "TES_limit_capn")
            C_inv_ = v.Q_TES_CAPn_L.sum() * p.c_TES_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["TES"] = C_inv_
            c.C_TOT_invAnn_["TES"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_TES_)
            c.C_TOT_RMI_["TES"] = C_inv_ * p.k_TES_RMI_

    def postprocess_func(self, r: Results):
        r.make_pos_ent("dQ_TES_in_TL", "dQ_TES_out_TL", "Storage output heat flow")


@dataclass
class H2H1(Component):
    """Heat downgrading from H2 to H1"""

    def param_func(self, sc: Scenario):
        sc.var("dQ_H2H1_T", doc="Heat down-grading", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_heating_sink_TH["H2H1"] = lambda t, h: v.dQ_H2H1_T[t] if h == 2 else 0
        c.dQ_heating_source_TH["H2H1"] = lambda t, h: v.dQ_H2H1_T[t] if h == 1 else 0


@dataclass
class BEV(Component):
    """Battery electric Vehicle"""

    E_CAPx: float = 100
    allow_V2X: bool = False
    allow_smart: bool = False

    def param_func(self, sc: Scenario):
        p = sc.params
        d = sc.dims
        sc.dim("B", data=[1, 2], doc="BEV batteries")
        sc.param("E_BEV_Cap1Bat_B", fill=self.E_CAPx, doc="Capacity of one battery", unit="kWh_el")
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
        sc.param(
            "eta_BEV_ch_",
            data=db.eta_BES_cycle_.data ** 0.5,
            doc="Charging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(
            "eta_BEV_dis_",
            data=db.eta_BES_cycle_.data ** 0.5,
            doc="Discharging efficiency",
            src="@Carroquino_2021",
        )
        sc.param("P_BEV_drive_TB", fill=0, doc="Power use", unit="kW_el")
        sc.param("y_BEV_avail_TB", fill=1, doc="If BEV is available for charging at time step")
        sc.param(
            "k_BEV_inPerCap_B",
            fill=0.7,
            doc="Maximum charging power per capacity",
            src="@Figgener_2021",
        )
        sc.param(
            "k_BEV_v2xPerCap_B",
            fill=0.7,
            doc="Maximum v2x discharging power per capacity",
            src="@Figgener_2021",
        )
        sc.param("k_BEV_empty_B", fill=0.05, doc="Minimum state of charge")
        sc.param("k_BEV_full_B", fill=0.95, doc="Maximum state of charge")
        sc.param("z_BEV_smart_", data=int(self.allow_smart), doc="If smart charging is allowed")
        sc.param("z_BEV_V2X_", data=int(self.allow_V2X), doc="If vehicle-to-X is allowed")
        sc.param("k_BEV_ini_B", fill=0.7, doc="Initial and final state of charge")
        sc.var("E_BEV_TB", doc="Electricity stored in BEV battery", unit="kWh_el")
        sc.var("P_BEV_in_TB", doc="Charging power", unit="kW_el")
        sc.var("P_BEV_V2X_TB", doc="Discharging power for vehicle-to-X", unit="kW_el")
        sc.var("x_BEV_penalty_", doc="Penalty to ensure uncontrolled charging in REF")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        T, B = d.T, d.B

        if sc.params.P_BEV_drive_TB.sum() == 0:
            logger.warning("P_BEV_drive_TB is all zero. Please set data")
        if sc.params.y_BEV_avail_TB.sum() == 1:
            logger.warning("y_BEV_avail_TB is all one. Please set data")

        m.addConstrs(
            (
                v.E_BEV_TB[t, b]
                == ((p.k_BEV_ini_B[b] * p.E_BEV_CAPx_B[b]) if t == T[0] else v.E_BEV_TB[t - 1, b])
                * p.eta_BEV_time_
                + p.k__dT_
                * (
                    v.P_BEV_in_TB[t, b] * p.eta_BEV_ch_
                    - (p.P_BEV_drive_TB[t, b] + v.P_BEV_V2X_TB[t, b]) / p.eta_BEV_dis_
                )
                for t in T
                for b in B
            ),
            "BEV_balance",
        )
        m.addConstrs(
            (v.E_BEV_TB[t, b] <= p.k_BEV_full_B[b] * p.E_BEV_CAPx_B[b] for t in T for b in B),
            "BEV_max_elec",
        )
        m.addConstrs(
            (v.E_BEV_TB[t, b] >= p.k_BEV_empty_B[b] * p.E_BEV_CAPx_B[b] for t in T for b in B),
            "BEV_min_elec",
        )

        m.addConstrs(
            (v.E_BEV_TB[T[-1], b] == p.k_BEV_ini_B[b] * p.E_BEV_CAPx_B[b] for b in B),
            "BEV_last_timestep",
        )
        m.addConstr(
            (
                v.x_BEV_penalty_
                == (1 - p.z_BEV_smart_) * quicksum(t * v.P_BEV_in_TB[t, b] for t in T for b in B)
            ),
            "BEV_penalty",
        )
        m.addConstrs(
            (
                v.P_BEV_in_TB[t, b]
                <= p.y_BEV_avail_TB[t, b] * p.E_BEV_CAPx_B[b] * p.k_BEV_inPerCap_B[b]
                for t in T
                for b in B
            ),
            "BEV_max_in",
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
            "BEV_max_V2X",
        )

        c.P_EL_source_T["BEV"] = lambda t: v.P_BEV_V2X_TB.sum(t, "*")
        c.P_EL_sink_T["BEV"] = lambda t: v.P_BEV_in_TB.sum(t, "*")
        c.Penalty_["BEV"] = v.x_BEV_penalty_


@dataclass
class pDem(Component):
    """Product demand and product balance

    Product balance:
        DEM <-- |
        PP  --> |
        PS  <-> |
    """

    sorts: Tuple = (1,)
    machines: Tuple = (1,)

    def dim_func(self, sc: Scenario):
        sc.dim("S", data=self.sorts, doc="Product sorts")
        sc.dim("M", data=self.machines, doc="Production machines")

    def param_func(self, sc: Scenario):
        sc.collector("dG_PROD_TS", doc="Positive = into the balance nod", unit="t/h")
        sc.param("dG_pDem_TS", fill=40, doc="Product demand", unit="t/h")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        T, S = d.T, d.S

        c.dG_PROD_TS["pDem"] = lambda t, s: -p.dG_pDem_TS[t, s]
        m.addConstrs(
            (quicksum(f(t, s) for f in c.dG_PROD_TS.values()) == 0 for t in T for s in S),
            "PROD_product_balance",
        )


@dataclass
class PP(Component):
    """Production process"""

    allow_new: bool = False

    def param_func(self, sc: Scenario):
        sc.param("c_PP_SU_", data=10, doc="Costs per start up", unit="€/SU")
        sc.param("c_PP_SC_", data=10, doc="Costs per sort change", unit="€/change")
        sc.param("y_PP_avail_TM", fill=1, doc="If avail")
        sc.param("y_PP_compat_SM", fill=1, doc="If machine and sort is compatible")
        sc.param("P_PP_CAPx_M", fill=2800, doc="", unit="kW_el")
        sc.param(
            "eta_PP_SM", fill=0.018, doc="Production efficiency", unit="t/kWh_el"
        )  # for cement mill 52–57 kWh/t or 0.017-0.020 t/kWh, see https://doi.org/10.1016/j.rser.2021.111963
        sc.param("k_PP_minPL_M", fill=1.0, doc="Minimum part load")
        sc.var("dG_PP_TSM", doc="Production of machine", unit="t/h")
        sc.var("P_PP_TSM", doc="Nominal power consumption of machine", unit="kW_el")
        sc.var("C_PP_SU_", doc="Total cost of start up", unit="k€")
        sc.var("C_PP_SC_", doc="Total cost of sort change", unit="k€")
        sc.var("Y_PP_op_TSM", doc="If machine is in operation", vtype=GRB.BINARY)
        sc.var("Y_PP_SU_TM", doc="If machine just started up", vtype=GRB.BINARY)
        sc.var("Y_PP_SC_TSM", doc="If sort hast just changed", vtype=GRB.BINARY)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        T, S, M = d.T, d.S, d.M

        m.addConstrs(
            (
                v.dG_PP_TSM[t, s, m]
                == v.P_PP_TSM[t, s, m]
                * p.eta_PP_SM[s, m]
                * p.y_PP_compat_SM[s, m]
                * p.y_PP_avail_TM[t, m]
                for t in T
                for s in S
                for m in M
            ),
            "PP_product_flow_balance",
        )
        m.addConstrs(
            (
                v.P_PP_TSM[t, s, m] <= v.Y_PP_op_TSM[t, s, m] * p.P_PP_CAPx_M[m]
                for t in T
                for s in S
                for m in M
            ),
            "PP_limit_power",
        )
        m.addConstrs(
            (
                v.P_PP_TSM[t, s, m] >= v.Y_PP_op_TSM[t, s, m] * p.k_PP_minPL_M[m] * p.P_PP_CAPx_M[m]
                for t in T
                for s in S
                for m in M
            ),
            "PP_minimum_part_load",
        )
        m.addConstr((v.C_PP_SU_ == v.Y_PP_SU_TM.sum() * p.c_PP_SU_), "PP_start_up")
        m.addConstrs(
            (
                v.Y_PP_SU_TM[t, m]
                >= v.Y_PP_op_TSM.sum(t, "*", m) - v.Y_PP_op_TSM.sum(t - 1, "*", m)
                for t in T[1:]
                for m in M
            ),
            "PP_start_up_2",
        )
        m.addConstr((v.C_PP_SC_ == v.Y_PP_SC_TSM.sum() * p.c_PP_SC_), "PP_sort_change")
        m.addConstrs(
            (
                v.Y_PP_SC_TSM[t, s, m] >= v.Y_PP_op_TSM[t, s, m] - v.Y_PP_op_TSM[t - 1, s, m]
                for t in T[1:]
                for m in M
                for s in S
            ),
            "PP_sort_change_2",
        )
        m.addConstrs(
            (v.dG_PP_TSM[t, s, m] <= v.Y_PP_op_TSM[t, s, m] * 1e8 for t in T for s in S for m in M),
            "PP_bigM",
        )
        m.addConstrs(
            (v.Y_PP_op_TSM.sum(t, "*", m) <= 1 for t in T for m in M),
            "PP_not_more_than_one_sort_per_machine",
        )

        c.dG_PROD_TS["PP"] = lambda t, s: v.dG_PP_TSM.sum(t, s, "*")
        c.P_EL_sink_T["PP"] = lambda t: v.P_PP_TSM.sum(t, "*", "*")
        c.C_TOT_op_["PP_SU"] = v.C_PP_SU_
        c.C_TOT_op_["PP_SC"] = v.C_PP_SC_


@dataclass
class PS(Component):
    """Product storage"""

    def param_func(self, sc: Scenario):
        sc.param("G_PS_CAPx_S", fill=2000, doc="Existing storage capacity of product", unit="t")
        sc.param("k_PS_min_S", fill=0.0, doc="Share of minimal required storage filling level")
        sc.param("k_PS_ini_S", fill=1.0, doc="Initial storage filling level")
        sc.var("G_PS_TS", doc="Storage filling level", unit="t")
        sc.var("G_PS_delta_S", doc="Final time step deviation from init", unit="t", lb=0)
        sc.var("E_PS_deltaTot_", doc="Energy equivalent", unit="kWh_el")

        if sc.consider_invest:
            sc.param("z_PS_S", fill=0, doc="If new storage capacity is allowed")
            sc.param("c_PS_inv_", data=1000, doc="Investment cost", unit="€/t")  # TODO
            sc.param("N_PS_", data=50, doc="Operation life", unit="a")
            sc.param("k_PS_RMI_", data=0)
            sc.var("G_PS_CAPn_S", doc="New capacity", unit="t")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        T, S = d.T, d.S

        def cap_PS_S(s):
            capx = p.G_PS_CAPx_S[s]
            return capx + v.G_PS_CAPn_S[s] if sc.consider_invest else capx

        m.addConstrs((v.G_PS_TS[t, s] <= cap_PS_S(s) for t in T for s in S), "RES_PS")
        m.addConstrs(
            (v.G_PS_TS[t, s] >= p.k_PS_min_S[s] * cap_PS_S(s) for t in T for s in S), "MIN_PS"
        )
        m.addConstrs(
            (v.G_PS_TS[T[-1], s] == p.k_PS_ini_S[s] * cap_PS_S(s) - v.G_PS_delta_S[s] for s in S),
            "PS_last_timestep",
        )
        m.addConstr(
            v.E_PS_deltaTot_ == v.G_PS_delta_S.sum() / sc.params.eta_PP_SM.min(), "PS_delta"
        )

        def get_output(t, s):
            if t == T[0]:
                return p.k_PS_ini_S[s] * cap_PS_S(s) - v.G_PS_TS[t, s] / p.k__dT_
            else:
                return v.G_PS_TS[t - 1, s] - v.G_PS_TS[t, s] / p.k__dT_

        c.dG_PROD_TS["PS"] = get_output
        c.Penalty_["PS"] = v.E_PS_deltaTot_ * sc.params.c_EG_T.mean()

        if sc.consider_invest:
            m.addConstrs((v.G_PS_CAPn_S[s] <= p.z_PS_S[s] * 1e9 for s in S), "PS_limit_capn")
            C_inv_ = v.G_PS_CAPn_S.sum() * p.c_PS_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["PS"] = C_inv_
            c.C_TOT_invAnn_["PS"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_PS_)
            c.C_TOT_RMI_["PS"] = C_inv_ * p.k_PS_RMI_


order_restrictions = [
    ("cDem", {}),
    ("hDem", {}),
    ("eDem", {}),
    ("EG", {"PV", "CHP"}),  # EG collects P_EG_sell_T
    ("Fuel", {"HOB", "CHP"}),  # Fuel collects F_fuel_F
    ("BES", {}),
    ("BEV", {}),
    ("PV", {}),
    ("P2H", {}),
    ("CHP", {}),
    ("HOB", {}),
    ("H2H1", {}),
    ("HP", {"cDem", "hDem"}),  # HP calculates COP based on thermal demand temperatures
    ("TES", {"cDem", "hDem"}),  # TESs can be defined for every thermal demand temperature level
    ("pDem", {"PP", "PS"}),
    ("PP", {}),
    ("PS", {}),
]
order_restrictions.append(("Main", [x[0] for x in order_restrictions]))

set_component_order_by_order_restrictions(order_restrictions=order_restrictions, classes=globals())
