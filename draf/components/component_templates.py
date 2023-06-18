import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum

from draf import Collectors, Dimensions, Params, Results, Scenario, Vars
from draf import helper as hp

# from draf.model_builder import autocollectors
from draf.abstract_component import Component
from draf.conventions import Descs
from draf.helper import conv, get_annuity_factor, set_component_order_by_order_restrictions
from draf.paths import DATA_DIR
from draf.prep import DataBase as db

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


@dataclass
class Main(Component):
    """Objective functions and general collectors. This must be the last model_func to be executed."""

    def param_func(self, sc: Scenario):
        sc.collector("P_EL_source_KG", doc="Power sources", unit="kW_el")
        sc.collector("P_EL_sink_KG", doc="Power sinks", unit="kW_el")
        sc.collector("dQ_cooling_source_KGN", doc="Cooling energy flow sources", unit="kW_th")
        sc.collector("dQ_cooling_sink_KGN", doc="Cooling energy flow sinks", unit="kW_th")
        sc.collector("dQ_heating_source_KGH", doc="Heating energy flow sources", unit="kW_th")
        sc.collector("dQ_heating_sink_KGH", doc="Heating energy flow sinks", unit="kW_th")
        sc.collector("dH_hydrogen_source_KG", doc="Hydrogen energy flow sources", unit="kW")
        sc.collector("dH_hydrogen_sink_KG", doc="Hydrogen energy flow sinks", unit="kW")
        sc.collector("C_TOT_", doc="Total costs", unit="k€/a")
        sc.collector("C_TOT_op_", doc="Total operating costs", unit="k€/a")
        sc.collector("CE_TOT_", doc="Total carbon emissions", unit="kgCO2eq/a")
        sc.collector("X_TOT_penalty_", doc="Penalty term for objective function", unit="Any")

        if sc.consider_invest:
            sc.collector("C_TOT_RMI_", doc="Total annual maintenance cost", unit="k€/a")
            sc.collector("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.collector("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")

        sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

        if sc.consider_invest:
            sc.param("k__r_", data=0.06, doc="Calculatory interest rate")
            sc.var("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.var("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")
            sc.var("C_TOT_RMI_", doc="Total annual maintenance cost", unit="k€")

        sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
        sc.param("k_PTO_C_", data=1, doc="Normalization factor")
        sc.param("k_PTO_CE_", data=1 / 1e4, doc="Normalization factor")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.setObjective(
            (
                (1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_
                + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_
                + quicksum(c.X_TOT_penalty_.values())
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
            c.C_TOT_op_["RMI"] = v.C_TOT_RMI_
            c.C_TOT_["inv"] = v.C_TOT_invAnn_

            # AUTOCOLLECTORS (currently unused) ---------------------------------------------------
            # m.addConstr( (v.C_TOT_inv_ == autocollectors.C_inv_(p, v, r=p.k__r_) * conv("€", "k€", 1e-3)) )
            # m.addConstr( (v.C_TOT_invAnn_ == autocollectors.C_inv_Annual_(p, v, r=p.k__r_) * conv("€", "k€", 1e-3)) )
            # m.addConstr( (v.C_TOT_RMI_ == autocollectors.C_TOT_RMI_(p, v) * conv("€", "k€", 1e-3)), "DEF_C_TOT_RMI_", )
            # -------------------------------------------------------------------------------------

        m.addConstr(v.C_TOT_ == quicksum(c.C_TOT_.values()), "total_cost_balance")
        m.addConstr(v.CE_TOT_ == quicksum(c.CE_TOT_.values()), "carbon_emission_balance")
        m.addConstrs(
            (
                quicksum(x(k, g) for x in c.P_EL_source_KG.values())
                == quicksum(x(k, g) for x in c.P_EL_sink_KG.values())
                for k in d.K
                for g in d.G
            ),
            "electricity_balance",
        )
        if c.dH_hydrogen_source_KG:
            m.addConstrs(
                (
                    quicksum(x(k, g) for x in c.dH_hydrogen_source_KG.values())
                    == quicksum(x(k, g) for x in c.dH_hydrogen_sink_KG.values())
                    for k in d.K
                    for g in d.G
                ),
                "hydrogen_balance",
            )

        if hasattr(d, "N"):
            m.addConstrs(
                (
                    quicksum(x(k, g, n) for x in c.dQ_cooling_source_KGN.values())
                    == quicksum(x(k, g, n) for x in c.dQ_cooling_sink_KGN.values())
                    for k in d.K
                    for g in d.G
                    for n in d.N
                ),
                "cool_balance",
            )

        if hasattr(d, "H"):
            m.addConstrs(
                (
                    quicksum(x(k, g, h) for x in c.dQ_heating_source_KGH.values())
                    == quicksum(x(k, g, h) for x in c.dQ_heating_sink_KGH.values())
                    for k in d.K
                    for g in d.G
                    for h in d.H
                ),
                "heat_balance",
            )


@dataclass
class cDem(Component):
    """Cooling demand"""

    def dim_func(self, sc: Scenario):
        sc.dim("N", data=["7/12", "30/35"], doc="Cooling temperature levels (inlet / outlet) in °C")

    def param_func(self, sc: Scenario):
        sc.param(name="dQ_cDem_KGN", fill=0, doc="Cooling demand", unit="kW_th")
        sc.params.dQ_cDem_KGN.loc[:, :, sc.dims.N[0]] = sc.prep.dQ_cDem_KG(
            annual_energy=1e4, set_param=False
        ).values
        sc.param(
            "T_cDem_in_N",
            data=[int(i.split("/")[0]) for i in sc.dims.N],
            doc="Cooling inlet temperature",
            unit="°C",
        )
        sc.param(
            "T_cDem_out_N",
            data=[int(i.split("/")[1]) for i in sc.dims.N],
            doc="Cooling outlet temperature",
            unit="°C",
        )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_cooling_source_KGN["cDem"] = lambda k, g, n: p.dQ_cDem_KGN[k, g, n]


@dataclass
class hDem(Component):
    """Heating demand"""

    def dim_func(self, sc: Scenario):
        sc.dim(
            "H", data=["90/60", "70/40"], doc="Heating temperature levels (inlet / outlet) in °C"
        )

    def param_func(self, sc: Scenario):
        sc.param(name="dQ_hDem_KGH", fill=0, doc="Heating demand", unit="kW_th")
        sc.params.dQ_hDem_KGH.loc[:, :, sc.dims.H[0]] = sc.prep.dQ_hDem_KG(
            annual_energy=1e6, set_param=False
        ).values
        sc.param(
            "T_hDem_in_H",
            data=[int(i.split("/")[0]) for i in sc.dims.H],
            doc="Heating inlet temperature",
            unit="°C",
        )
        sc.param(
            "T_hDem_out_H",
            data=[int(i.split("/")[1]) for i in sc.dims.H],
            doc="Heating outlet temperature",
            unit="°C",
        )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_heating_sink_KGH["hDem"] = lambda k, g, h: p.dQ_hDem_KGH[k, g, h]


@dataclass
class eDem(Component):
    """Electricity demand"""

    p_el: Optional[pd.Series] = None
    profile: str = "G3"
    annual_energy: float = 5e6

    def param_func(self, sc: Scenario):
        if self.p_el is None:
            sc.prep.P_eDem_KG(profile=self.profile, annual_energy=self.annual_energy)
        else:
            sc.param("P_eDem_KG", data=self.p_el, doc="Electricity demand", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.P_EL_sink_KG["eDem"] = lambda k, g: p.P_eDem_KG[k, g]


@dataclass
class EG(Component):
    """Electricity grid"""

    c_buyPeak: float = 50.0
    prepared_tariffs: Tuple = ("FLAT", "TOU", "RTP")
    selected_tariff: str = "RTP"
    feedin_reduces_emissions: bool = False
    maxsell: float = 20e3
    maxbuy: float = 20e3

    def param_func(self, sc: Scenario):
        sc.collector("P_EG_sell_KG", doc="Sold electricity power", unit="kW_el")
        sc.param("c_EG_buyPeak_", data=self.c_buyPeak, doc="Peak price", unit="€/kW_el/a")

        if "RTP" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_KG()
        if "TOU" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_KG()
            sc.prep.c_EG_TOU_KG()
        if "FLAT" in self.prepared_tariffs:
            sc.prep.c_EG_RTP_KG()
            sc.prep.c_EG_FLAT_KG()
        sc.param(
            "c_EG_KG",
            data=getattr(sc.params, f"c_EG_{self.selected_tariff}_KG"),
            doc="Chosen electricity tariff",
            unit="€/kWh_el",
        )
        sc.prep.c_EG_addon_()
        sc.prep.ce_EG_KG()
        sc.param("t_EG_minFLH_", data=0, doc="Minimal full load hours", unit="h")
        sc.var("P_EG_buy_KG", doc="Purchased electrical power", unit="kW_el")
        sc.var("P_EG_sell_KG", doc="Selling electrical power", unit="kW_el", ub=self.maxsell)
        sc.var("P_EG_buyPeak_", doc="Peak electrical power", unit="kW_el", ub=self.maxbuy)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.P_EG_sell_KG[k, g] == sum(x(k, g) for x in c.P_EG_sell_KG.values())
                for k in d.K
                for g in d.G
            ),
            "EG_sell",
        )
        m.addConstrs(
            (v.P_EG_buy_KG[k, g] <= v.P_EG_buyPeak_ for k in d.K for g in d.G), "EG_peak_price"
        )

        if p.t_EG_minFLH_ > 0:
            m.addConstr(
                quicksum(
                    v.P_EG_buy_KG[k, g] * sc.dt(k, g) * sc.periodOccurrences[k]
                    for k in d.K
                    for g in d.G
                )
                >= p.t_EG_minFLH_ * v.P_EG_buyPeak_,
                "EG_minimum_full_load_hours",
            )

        c.P_EL_source_KG["EG"] = lambda k, g: v.P_EG_buy_KG[k, g]
        c.P_EL_sink_KG["EG"] = lambda k, g: v.P_EG_sell_KG[k, g]
        c.C_TOT_op_["EG_peak"] = v.P_EG_buyPeak_ * p.c_EG_buyPeak_ * conv("€", "k€", 1e-3)
        c.C_TOT_op_["EG"] = quicksum(
            (
                v.P_EG_buy_KG[k, g] * (p.c_EG_KG[k, g] + p.c_EG_addon_)
                - v.P_EG_sell_KG[k, g] * p.c_EG_KG[k, g]
            )
            * sc.dt(k, g)
            * sc.periodOccurrences[k]
            for k in d.K
            for g in d.G
        ) * conv("€", "k€", 1e-3)
        if self.feedin_reduces_emissions:
            c.CE_TOT_["EG"] = quicksum(
                p.ce_EG_KG[k, g]
                * (v.P_EG_buy_KG[k, g] - v.P_EG_sell_KG[k, g])
                * sc.dt(k, g)
                * sc.periodOccurrences[k]
                for k in d.K
                for g in d.G
            )
        else:
            c.CE_TOT_["EG"] = quicksum(
                p.ce_EG_KG[k, g] * v.P_EG_buy_KG[k, g] * sc.dt(k, g) * sc.periodOccurrences[k]
                for k in d.K
                for g in d.G
            )

    def postprocess_func(self, sc: Scenario):
        sc.res.make_pos_ent("P_EG_buy_KG")


@dataclass
class Fuel(Component):
    """Fuels"""

    c_ceTax: float = 55

    def dim_func(self, sc: Scenario):
        sc.dim("F", ["ng", "bio"], doc="Types of fuel")

    def param_func(self, sc: Scenario):
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
        m.addConstr(v.CE_Fuel_ == quicksum(v.F_fuel_F[f] * p.ce_Fuel_F[f] for f in d.F))
        m.addConstr(
            v.C_Fuel_
            == quicksum(v.F_fuel_F[f] * p.c_Fuel_F[f] for f in d.F) * conv("€", "k€", 1e-3)
        )
        m.addConstr(
            v.C_Fuel_ceTax_
            == p.c_Fuel_ceTax_ * conv("/t", "(/kg", 1e-3) * v.CE_Fuel_ * conv("€", "k€", 1e-3)
        )
        c.CE_TOT_["Fuel"] = v.CE_Fuel_
        c.C_TOT_op_["Fuel"] = v.C_Fuel_
        c.C_TOT_op_["FuelCeTax"] = v.C_Fuel_ceTax_


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
            data=db.eta_BES_cycle_.data**0.5,
            doc="Charging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(
            "eta_BES_dis_",
            data=db.eta_BES_cycle_.data**0.5,
            doc="Discharging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(from_db=db.eta_BES_self_)
        sc.param(from_db=db.k_BES_inPerCap_)
        sc.param(from_db=db.k_BES_outPerCap_)
        sc.var("E_BES_KG", doc="Electricity stored", unit="kWh_el")
        sc.var("P_BES_in_KG", doc="Charging power", unit="kW_el")
        sc.var("P_BES_out_KG", doc="Discharging power", unit="kW_el")

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
            (v.P_BES_in_KG[k, g] <= p.k_BES_inPerCap_ * cap for k in d.K for g in d.G),
            "BES_limit_charging_power",
        )
        m.addConstrs(
            (v.P_BES_out_KG[k, g] <= p.k_BES_outPerCap_ * cap for k in d.K for g in d.G),
            "BES_limit_discharging_power",
        )
        m.addConstrs((v.E_BES_KG[k, g] <= cap for k in d.K for g in d.G), "BES_limit_cap")
        m.addConstrs(
            (v.E_BES_KG[k, d.G[-1]] == p.k_BES_ini_ * cap for k in d.K), "BES_last_timestep"
        )
        m.addConstrs(
            (
                v.E_BES_KG[k, g]
                == (p.k_BES_ini_ * cap if g == d.G[0] else v.E_BES_KG[k, g - 1])
                * (1 - p.eta_BES_self_ * sc.dt(k, g))
                + (v.P_BES_in_KG[k, g] * p.eta_BES_ch_ - v.P_BES_out_KG[k, g] / p.eta_BES_dis_)
                * sc.dt(k, g)
                for k in d.K
                for g in d.G
            ),
            "BES_electricity_balance",
        )
        c.P_EL_source_KG["BES"] = lambda k, g: v.P_BES_out_KG[k, g]
        c.P_EL_sink_KG["BES"] = lambda k, g: v.P_BES_in_KG[k, g]
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
        sc.prep.P_PV_profile_KG(use_coords=True)
        sc.var("P_PV_FI_KG", doc="Feed-in", unit="kW_el")
        sc.var("P_PV_OC_KG", doc="Own consumption", unit="kW_el")
        sc.param(
            "A_PV_PerPeak_",
            data=6.5,
            doc="Area efficiency of new PV",
            unit="m²/kW_peak",
            src="https://www.dachvermieten.net/wieviel-qm-dachflaeche-fuer-1-kw-kilowatt",
        )
        sc.param("A_PV_avail_", data=self.A_avail_, doc="Area available for new PV", unit="m²")

        if sc.consider_invest:
            sc.param("z_PV_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_PV_inv_())
            sc.param(from_db=db.k_PV_RMI_)
            sc.param(from_db=db.N_PV_)
            sc.var("P_PV_CAPn_", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.P_PV_CAPx_ + v.P_PV_CAPn_ if sc.consider_invest else p.P_PV_CAPx_
        m.addConstrs(
            (
                cap * p.P_PV_profile_KG[k, g] == v.P_PV_FI_KG[k, g] + v.P_PV_OC_KG[k, g]
                for k in d.K
                for g in d.G
            ),
            "PV_balance",
        )
        c.P_EL_source_KG["PV"] = lambda k, g: v.P_PV_FI_KG[k, g] + v.P_PV_OC_KG[k, g]
        c.P_EG_sell_KG["PV"] = lambda k, g: v.P_PV_FI_KG[k, g]

        if sc.consider_invest:
            m.addConstr(v.P_PV_CAPn_ <= p.z_PV_ * p.A_PV_avail_ / p.A_PV_PerPeak_, "PV_limit_capn")
            C_inv_ = v.P_PV_CAPn_ * p.c_PV_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["PV"] = C_inv_
            c.C_TOT_invAnn_["PV"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_PV_)
            c.C_TOT_RMI_["PV"] = C_inv_ * p.k_PV_RMI_


@dataclass
class WT(Component):
    """Wind turbine"""

    P_CAPx: float = 0
    allow_new: bool = True
    pay_network_tariffs: bool = True

    def param_func(self, sc: Scenario):
        sc.param("P_WT_CAPx_", data=self.P_CAPx, doc="Existing capacity", unit="kW_peak")
        sc.param(
            "P_WT_profile_KG",
            data=hp.read(DATA_DIR / "wind/2019_wind_kelmarsh2.csv"),
            doc="Wind profile",
            unit="kW_el",
        )
        sc.param(
            "y_WT_pnt_",
            data=int(self.pay_network_tariffs),
            doc="If `c_EG_addon_` is paid on own wind energy consumption (e.g. for off-site PPA)",
        )
        sc.var("P_WT_FI_KG", doc="Feed-in", unit="kW_el")
        sc.var("P_WT_OC_KG", doc="Own consumption", unit="kW_el")

        if sc.consider_invest:
            sc.param("P_WT_max_", data=1e5, doc="Maximum installed capacity", unit="kW_peak")
            sc.param("z_WT_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(
                "c_WT_inv_",
                data=1682,
                doc="CAPEX",
                unit="€/kW_peak",
                src="https://windeurope.org/newsroom/press-releases/europe-invested-41-bn-euros-in-new-wind-farms-in-2021",
            )  # or 1118.77 €/kWp invest and 27 years operation life for onshore wind https://github.com/PyPSA/technology-data/blob/4eaddec90f429246445f08476b724393dde753c8/outputs/costs_2020.csv
            sc.param(
                "k_WT_RMI_",
                data=0.01,
                doc=Descs.RMI.en,
                unit="",
                src="https://www.npro.energy/main/en/help/economic-parameters",
            )
            sc.param(
                "N_WT_",
                data=20,
                doc="Operation life",
                unit="a",
                src="https://www.twi-global.com/technical-knowledge/faqs/how-long-do-wind-turbines-last",
            )
            sc.var("P_WT_CAPn_", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.P_WT_CAPx_ + v.P_WT_CAPn_ if sc.consider_invest else p.P_WT_CAPx_
        m.addConstrs(
            (
                cap * p.P_WT_profile_KG[k, g] == v.P_WT_FI_KG[k, g] + v.P_WT_OC_KG[k, g]
                for k in d.K
                for g in d.G
            ),
            "WT_balance",
        )
        c.P_EL_source_KG["WT"] = lambda k, g: v.P_WT_FI_KG[k, g] + v.P_WT_OC_KG[k, g]
        c.P_EG_sell_KG["WT"] = lambda k, g: v.P_WT_FI_KG[k, g]

        if p.y_WT_pnt_:
            c.C_TOT_op_["WT"] = (
                quicksum(
                    v.P_WT_OC_KG[k, g] * sc.dt(k, g) * sc.periodOccurrences[k]
                    for k in d.K
                    for g in d.G
                )
                * p.c_EG_addon_
                * conv("€", "k€", 1e-3)
            )

        if sc.consider_invest:
            m.addConstr(v.P_WT_CAPn_ <= p.z_WT_ * p.P_WT_max_, "WT_limit_capn")
            C_inv_ = v.P_WT_CAPn_ * p.c_WT_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["WT"] = C_inv_
            c.C_TOT_invAnn_["WT"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_WT_)
            c.C_TOT_RMI_["WT"] = C_inv_ * p.k_WT_RMI_


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

    def dim_func(self, sc: Scenario):
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

    def param_func(self, sc: Scenario):
        p = sc.params

        sc.collector("dQ_amb_source_", doc="Thermal energy flow to ambient", unit="kW_th")
        sc.collector("dQ_amb_sink_", doc="Thermal energy flow from ambient", unit="kW_th")

        if self.time_dependent_amb:
            sc.prep.T__amb_KG()
        sc.param("T__amb_", data=25, doc="Approximator for ambient air", unit="°C")

        sc.param(
            "T_HP_Cond_C", data=p.T_hDem_in_H + 5, doc="Condensation side temperature", unit="°C"
        )
        sc.param(
            "T_HP_Eva_E", data=p.T_cDem_in_N - 5, doc="Evaporation side temperature", unit="°C"
        )
        sc.param("n_HP_", data=self.n, doc="Maximum number of parallel operation modes")
        sc.param(
            "eta_HP_",
            data=0.5,
            doc="Ratio of reaching the ideal COP (exergy efficiency)",
            src="@Arat_2017",
            # Cox_2022 used 0.45: https://doi.org/10.1016/j.apenergy.2021.118499
            # but roughly 0.5 in recent real operation of high temperature HP: https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/01_Verband/Webinare/Vortrag_Wilk_AIT_02062020.pdf
        )
        sc.param("dQ_HP_CAPx_", data=self.dQ_CAPx, doc="Existing heating capacity", unit="kW_th")
        sc.param(
            "dQ_HP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_th"
        )
        sc.var("P_HP_KGEC", doc="Consuming power", unit="kW_el")
        sc.var("dQ_HP_Cond_KGEC", doc="Heat flow released on condensation side", unit="kW_th")
        sc.var("dQ_HP_Eva_KGEC", doc="Heat flow absorbed on evaporation side", unit="kW_th")
        sc.var("Y_HP_KGEC", doc="If source and sink are connected at time-step", vtype=GRB.BINARY)

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
        def get_cop(k, g, e, c):
            T_amb = p.T__amb_KG[k, g] if self.time_dependent_amb else p.T__amb_
            T_cond = T_amb + 5 if c == "C_amb" else p.T_HP_Cond_C[c]
            T_eva = T_amb - 5 if e == "E_amb" else p.T_HP_Eva_E[e]
            return 100 if T_cond <= T_eva else p.eta_HP_ * (T_cond + 273) / (T_cond - T_eva)

        m.addConstrs(
            (
                v.dQ_HP_Cond_KGEC[k, g, e, c] == v.P_HP_KGEC[k, g, e, c] * get_cop(k, g, e, c)
                for k in d.K
                for g in d.G
                for e in d.E
                for c in d.C
            ),
            "HP_balance_1",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_KGEC[k, g, e, c]
                == v.dQ_HP_Eva_KGEC[k, g, e, c] + v.P_HP_KGEC[k, g, e, c]
                for k in d.K
                for g in d.G
                for e in d.E
                for c in d.C
            ),
            "HP_balance_2",
        )
        m.addConstrs(
            (
                v.dQ_HP_Cond_KGEC[k, g, e, c] <= v.Y_HP_KGEC[k, g, e, c] * p.dQ_HP_max_
                for k in d.K
                for g in d.G
                for e in d.E
                for c in d.C
            ),
            "HP_bigM",
        )
        if ("E_amb" in d.E) and ("C_amb" in d.C):
            # Avoid E_amb --> C_amb HP operation occur due to negative electricity prices
            m.addConstr((v.Y_HP_KGEC.sum("*", "E_amb", "C_amb") == 0), "HP_no_Eamb_to_C_amb")
        cap = p.dQ_HP_CAPx_ + v.dQ_HP_CAPn_ if sc.consider_invest else p.dQ_HP_CAPx_
        m.addConstrs(
            (v.dQ_HP_Cond_KGEC.sum(k, g, "*", "*") <= cap for k in d.K for g in d.G), "HP_limit_cap"
        )
        m.addConstrs(
            (v.Y_HP_KGEC.sum(k, g, "*", "*") <= p.n_HP_ for k in d.K for g in d.G),
            "HP_operating_mode",
        )

        c.P_EL_sink_KG["HP"] = lambda k, g: v.P_HP_KGEC.sum(k, g, "*", "*")
        c.dQ_cooling_sink_KGN["HP"] = lambda k, g, n: v.dQ_HP_Eva_KGEC.sum(k, g, n, "*")
        c.dQ_heating_source_KGH["HP"] = lambda k, g, h: v.dQ_HP_Cond_KGEC.sum(k, g, "*", h)
        c.dQ_amb_sink_["HP"] = quicksum(
            v.dQ_HP_Eva_KGEC[k, g, "E_amb", c] * sc.dt(k, g) * sc.periodOccurrences[k]
            for k in d.K
            for g in d.G
            for c in d.C
        )
        c.dQ_amb_source_["HP"] = quicksum(
            v.dQ_HP_Cond_KGEC[k, g, e, "C_amb"] * sc.dt(k, g) * sc.periodOccurrences[k]
            for k in d.K
            for g in d.G
            for e in d.E
        )

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
    H_level_target: str = "90/60"

    def param_func(self, sc: Scenario):
        sc.param("dQ_P2H_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param(from_db=db.eta_P2H_)
        sc.var("P_P2H_KG", doc="Consuming power", unit="kW_el")
        sc.var("dQ_P2H_KG", doc="Producing heat flow", unit="kW_th")

        if sc.consider_invest:
            sc.param(from_db=db.N_P2H_)
            sc.param("z_P2H_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.c_P2H_inv_)
            sc.param("k_P2H_RMI_", data=0, doc=Descs.RMI.en)
            sc.var("dQ_P2H_CAPn_", doc="New capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.dQ_P2H_CAPx_ + v.dQ_P2H_CAPn_ if sc.consider_invest else p.dQ_P2H_CAPx_
        m.addConstrs(
            (v.dQ_P2H_KG[k, g] == p.eta_P2H_ * v.P_P2H_KG[k, g] for k in d.K for g in d.G),
            "P2H_balance",
        )
        m.addConstrs((v.dQ_P2H_KG[k, g] <= cap for k in d.K for g in d.G), "P2H_limit_heat_flow")

        c.dQ_heating_source_KGH["P2H"] = (
            lambda k, g, h: v.dQ_P2H_KG[k, g] if h == self.H_level_target else 0
        )
        c.P_EL_sink_KG["P2H"] = lambda k, g: v.P_P2H_KG[k, g]

        if sc.consider_invest:
            m.addConstr((v.dQ_P2H_CAPn_ <= p.z_P2H_ * 1e6), "P2H_limit_new_capa")
            C_inv_ = v.dQ_P2H_CAPn_ * p.c_P2H_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["P2H"] = C_inv_
            c.C_TOT_invAnn_["P2H"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_P2H_)
            c.C_TOT_RMI_["P2H"] = C_inv_ * p.k_P2H_RMI_


@dataclass
class DAC(Component):
    """Direct air capture: More precisely carbon offsetting through direct air carbon capture
    and storage. Reduces total carbon emissions for a price per unit of carbon emissions.
    """

    allow_new: bool = True

    def param_func(self, sc: Scenario):
        sc.param("z_DAC_", data=int(self.allow_new), doc="If DAC is allowed")
        sc.param(
            "c_DAC_",
            data=222,
            doc="Cost of direct air capture and storage",
            unit="€/tCO2eq",
            src="https://doi.org/10.1016/j.jclepro.2019.03.086",
        )
        sc.var("CE_DAC_", doc="Carbon emissions captured and stored by DAC", unit="kgCO2eq/a")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.C_TOT_op_["DAC"] = v.CE_DAC_ * p.c_DAC_ * conv("€", "k€", 1e-3) * conv("/t", "/kg", 1e-3)
        c.CE_TOT_["DAC"] = -v.CE_DAC_


@dataclass
class CHP(Component):
    """Combined heat and power"""

    P_CAPx: float = 0
    allow_new: bool = True
    H_level_target: str = "90/60"
    minPL: Optional[float] = 0.5

    def param_func(self, sc: Scenario):
        sc.param("P_CHP_CAPx_", data=self.P_CAPx, doc="Existing capacity", unit="kW_el")
        sc.param(
            "P_CHP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_el"
        )
        sc.param(from_db=db.funcs.eta_CHP_el_(fuel="ng"))
        sc.param(from_db=db.funcs.eta_CHP_th_(fuel="ng"))
        sc.var("dQ_CHP_KG", doc="Producing heat flow", unit="kW_th")
        sc.var("F_CHP_KGF", doc="Consumed fuel flow", unit="kW")
        sc.var("P_CHP_FI_KG", doc="Feed-in", unit="kW_el")
        sc.var("P_CHP_OC_KG", doc="Own consumption", unit="kW_el")
        sc.var("P_CHP_KG", doc="Producing power", unit="kW_el")

        if self.minPL is not None:
            sc.param("k_CHP_minPL_", data=0.5, doc="Minimal allowed part load")
            sc.var("Y_CHP_KG", doc="If in operation", vtype=GRB.BINARY)

        if sc.consider_invest:
            sc.param("z_CHP_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_CHP_inv_(estimated_size=400, fuel_type="ng"))
            sc.param(from_db=db.k_CHP_RMI_)
            sc.param(from_db=db.N_CHP_)
            sc.var("P_CHP_CAPn_", doc="New capacity", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.P_CHP_KG[k, g] == p.eta_CHP_el_ * quicksum(v.F_CHP_KGF[k, g, f] for f in d.F)
                for k in d.K
                for g in d.G
            ),
            "CHP_elec_balance",
        )
        m.addConstrs(
            (
                v.dQ_CHP_KG[k, g] == p.eta_CHP_th_ * quicksum(v.F_CHP_KGF[k, g, f] for f in d.F)
                for k in d.K
                for g in d.G
            ),
            "CHP_heat_balance",
        )
        cap = p.P_CHP_CAPx_ + v.P_CHP_CAPn_ if sc.consider_invest else p.P_CHP_CAPx_
        m.addConstrs((v.P_CHP_KG[k, g] <= cap for k in d.K for g in d.G), "CHP_limit_elecPower")
        m.addConstrs(
            (
                v.P_CHP_KG[k, g] == v.P_CHP_FI_KG[k, g] + v.P_CHP_OC_KG[k, g]
                for k in d.K
                for g in d.G
            ),
            "CHP_feedIn_vs_ownConsumption",
        )

        if self.minPL:
            m.addConstrs(
                (v.P_CHP_KG[k, g] <= v.Y_CHP_KG[k, g] * p.P_CHP_max_ for k in d.K for g in d.G),
                "CHP_minimal_part_load_1",
            )
            m.addConstrs(
                (
                    v.P_CHP_KG[k, g] >= p.k_CHP_minPL_ * cap - p.P_CHP_max_ * (1 - v.Y_CHP_KG[k, g])
                    for k in d.K
                    for g in d.G
                ),
                "CHP_minimal_part_load_2",
            )

        c.P_EL_source_KG["CHP"] = lambda k, g: v.P_CHP_FI_KG[k, g] + v.P_CHP_OC_KG[k, g]
        c.dQ_heating_source_KGH["CHP"] = (
            lambda k, g, h: v.dQ_CHP_KG[k, g] if h == self.H_level_target else 0
        )
        c.P_EG_sell_KG["CHP"] = lambda k, g: v.P_CHP_FI_KG[k, g]
        c.F_fuel_F["CHP"] = lambda f: quicksum(
            v.F_CHP_KGF[k, g, f] * sc.dt(k, g) * sc.periodOccurrences[k] for k in d.K for g in d.G
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
    allow_new: bool = True
    H_level_target: str = "90/60"

    def param_func(self, sc: Scenario):
        sc.param("dQ_HOB_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_th")
        sc.param("eta_HOB_", from_db=db.eta_HOB_)
        sc.var("dQ_HOB_KG", doc="Ouput heat flow", unit="kW_th")
        sc.var("F_HOB_KGF", doc="Input fuel flow", unit="kW")

        if sc.consider_invest:
            sc.param(from_db=db.funcs.c_HOB_inv_())
            sc.param(from_db=db.k_HOB_RMI_)
            sc.param(from_db=db.N_HOB_)
            sc.param("z_HOB_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.var("dQ_HOB_CAPn_", doc="New capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.dQ_HOB_CAPx_ + v.dQ_HOB_CAPn_ if sc.consider_invest else p.dQ_HOB_CAPx_
        m.addConstrs(
            (
                v.dQ_HOB_KG[k, g] == v.F_HOB_KGF.sum(k, g, "*") * p.eta_HOB_
                for k in d.K
                for g in d.G
            ),
            "HOB_bal",
        )
        m.addConstrs((v.dQ_HOB_KG[k, g] <= cap for k in d.K for g in d.G), "HOB_limit_heat_flow")

        c.dQ_heating_source_KGH["HOB"] = (
            lambda k, g, h: v.dQ_HOB_KG[k, g] if h == self.H_level_target else 0
        )
        c.F_fuel_F["HOB"] = lambda f: quicksum(
            v.F_HOB_KGF[k, g, f] * sc.dt(k, g) * sc.periodOccurrences[k] for k in d.K for g in d.G
        )

        if sc.consider_invest:
            m.addConstr((v.dQ_HOB_CAPn_ <= p.z_HOB_ * 1e6), "HOB_limit_new_capa")
            C_inv_ = v.dQ_HOB_CAPn_ * p.c_HOB_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["HOB"] = C_inv_
            c.C_TOT_invAnn_["HOB"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_HOB_)
            c.C_TOT_RMI_["HOB"] = C_inv_ * p.k_HOB_RMI_


@dataclass
class TES(Component):
    """Thermal energy storage"""

    allow_new: bool = True

    def dim_func(self, sc: Scenario):
        d = sc.dims

        L = []
        if hasattr(d, "N"):
            L += d.N
        if hasattr(d, "H"):
            L += d.H
        sc.dim("L", data=L, doc="Thermal demand temperature levels (inlet / outlet) in °C")

    def param_func(self, sc: Scenario):
        sc.param("Q_TES_CAPx_L", fill=0, doc="Existing capacity", unit="kWh_th")
        sc.param("eta_TES_self_", data=0.005, doc="Self-discharge")
        sc.param("k_TES_inPerCap_", data=0.5, doc="Ratio loading power / capacity")
        sc.param("k_TES_outPerCap_", data=0.5, doc="Ratio unloading power / capacity")
        sc.param("k_TES_ini_L", fill=0.5, doc="Initial and final energy level share")
        sc.var("dQ_TES_in_KGL", doc="Storage input heat flow", unit="kW_th", lb=-GRB.INFINITY)
        sc.var("Q_TES_KGL", doc="Stored heat", unit="kWh_th")

        if sc.consider_invest:
            sc.param("z_TES_L", fill=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.funcs.c_TES_inv_(estimated_size=100, temp_spread=40))
            sc.param(from_db=db.k_TES_RMI_)
            sc.param(from_db=db.N_TES_)
            sc.var("Q_TES_CAPn_L", doc="New capacity", unit="kWh_th", ub=1e7)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        def cap(l):
            return p.Q_TES_CAPx_L[l] + (
                v.Q_TES_CAPn_L[l] if sc.consider_invest else p.Q_TES_CAPx_L[l]
            )

        m.addConstrs(
            (
                v.Q_TES_KGL[k, g, l]
                == ((p.k_TES_ini_L[l] * cap(l)) if g == d.G[0] else v.Q_TES_KGL[k, g - 1, l])
                * (1 - p.eta_TES_self_ * sc.dt(k, g))
                + sc.dt(k, g) * v.dQ_TES_in_KGL[k, g, l]
                for k in d.K
                for g in d.G
                for l in d.L
            ),
            "TES_balance",
        )
        m.addConstrs(
            (v.Q_TES_KGL[k, g, l] <= cap(l) for k in d.K for g in d.G for l in d.L), "TES_limit_cap"
        )
        m.addConstrs(
            (
                v.dQ_TES_in_KGL[k, g, l] <= p.k_TES_inPerCap_ * cap(l)
                for k in d.K
                for g in d.G
                for l in d.L
            ),
            "TES_limit_in",
        )
        m.addConstrs(
            (
                v.dQ_TES_in_KGL[k, g, l] >= -p.k_TES_outPerCap_ * cap(l)
                for k in d.K
                for g in d.G
                for l in d.L
            ),
            "TES_limit_out",
        )
        m.addConstrs(
            (v.Q_TES_KGL[k, d.G[-1], l] == p.k_TES_ini_L[l] * cap(l) for k in d.K for l in d.L),
            "TES_last_timestep",
        )

        # only sink here, since dQ_TES_in_KGL is also defined for negative
        # values to reduce number of variables:
        if hasattr(d, "N"):
            c.dQ_cooling_sink_KGN["TES"] = lambda k, g, n: v.dQ_TES_in_KGL[k, g, n]
        if hasattr(d, "H"):
            c.dQ_heating_sink_KGH["TES"] = lambda k, g, h: v.dQ_TES_in_KGL[k, g, h]

        if sc.consider_invest:
            m.addConstrs((v.Q_TES_CAPn_L[l] <= p.z_TES_L[l] * 1e5 for l in d.L), "TES_limit_capn")
            C_inv_ = v.Q_TES_CAPn_L.sum() * p.c_TES_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["TES"] = C_inv_
            c.C_TOT_invAnn_["TES"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_TES_)
            c.C_TOT_RMI_["TES"] = C_inv_ * p.k_TES_RMI_

    def postprocess_func(self, sc: Scenario):
        sc.res.make_pos_ent("dQ_TES_in_KGL", "dQ_TES_out_KGL", "Storage output heat flow")


@dataclass
class HD(Component):
    """Heat downgrading"""

    from_level: str = "90/60"
    to_level: str = "70/40"

    def param_func(self, sc: Scenario):
        sc.var("dQ_HD_KG", doc="Heat down-grading", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dQ_heating_sink_KGH["HD"] = (
            lambda k, g, h: v.dQ_HD_KG[k, g] if h == self.from_level else 0
        )
        c.dQ_heating_source_KGH["HD"] = (
            lambda k, g, h: v.dQ_HD_KG[k, g] if h == self.to_level else 0
        )


@dataclass
class BEV(Component):
    """Battery electric Vehicle"""

    allow_V2X: bool = False
    allow_smart: bool = False

    def dim_func(self, sc: Scenario):
        sc.dim("B", data=[1, 2], doc="BEV batteries")

    def param_func(self, sc: Scenario):
        p = sc.params
        sc.param("E_BEV_CAPx_B", fill=1000, doc="Capacity of all batteries", unit="kWh_el")
        sc.param(
            "eta_BEV_self_",
            data=0.0,
            doc="Self discharge. Must be 0.0 for the uncontrolled charging in REF",
        )
        sc.param(
            "eta_BEV_ch_",
            data=db.eta_BES_cycle_.data**0.5,
            doc="Charging efficiency",
            src="@Carroquino_2021",
        )
        sc.param(
            "eta_BEV_dis_",
            data=db.eta_BES_cycle_.data**0.5,
            doc="Discharging efficiency",
            src="@Carroquino_2021",
        )
        sc.param("P_BEV_drive_KGB", fill=0, doc="Power use", unit="kW_el")
        sc.param("y_BEV_avail_KGB", fill=1, doc="If BEV is available for charging at time step")
        sc.param(
            "k_BEV_inPerCap_B",
            fill=0.7,
            doc="Maximum charging power per capacity",
            src="@Figgener_2021",
        )  # NOTE: Similar for powered industrial trucks: "a 25 V lithium-ion battery is fully charged in only 80 minutes" https://www.mfgabelstapler.de/2019/02/22/vorteile-von-lithium-ionen-batterien-fuer-gabelstapler
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
        sc.var("E_BEV_KGB", doc="Electricity stored in BEV battery", unit="kWh_el")
        sc.var("P_BEV_in_KGB", doc="Charging power", unit="kW_el")
        sc.var("P_BEV_V2X_KGB", doc="Discharging power for vehicle-to-X", unit="kW_el")
        sc.var("X_BEV_penalty_", doc="Penalty to ensure uncontrolled charging in REF")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        B, G, K = d.B, d.G, d.K

        if sc.params.P_BEV_drive_KGB.sum() == 0:
            logger.warning("P_BEV_drive_KGB is all zero. Please set data")
        if sc.params.y_BEV_avail_KGB.mean() == 1:
            logger.warning("y_BEV_avail_KGB is all one. Please set data")
        m.addConstrs(
            (
                v.E_BEV_KGB[k, g, b]
                == (
                    (p.k_BEV_ini_B[b] * p.E_BEV_CAPx_B[b])
                    if g == G[0]
                    else v.E_BEV_KGB[k, g - 1, b]
                )
                * (1 - p.eta_BEV_self_ * sc.dt(k, g))
                + sc.dt(k, g)
                * (
                    v.P_BEV_in_KGB[k, g, b] * p.eta_BEV_ch_
                    - (p.P_BEV_drive_KGB[k, g, b] + v.P_BEV_V2X_KGB[k, g, b]) / p.eta_BEV_dis_
                )
                for k in K
                for g in G
                for b in B
            ),
            "BEV_balance",
        )
        m.addConstrs(
            (
                v.E_BEV_KGB[k, g, b] <= p.k_BEV_full_B[b] * p.E_BEV_CAPx_B[b]
                for k in K
                for g in G
                for b in B
            ),
            "BEV_max_elec",
        )
        m.addConstrs(
            (
                v.E_BEV_KGB[k, g, b] >= p.k_BEV_empty_B[b] * p.E_BEV_CAPx_B[b]
                for k in K
                for g in G
                for b in B
            ),
            "BEV_min_elec",
        )

        m.addConstrs(
            (
                v.E_BEV_KGB[k, G[-1], b] == p.k_BEV_ini_B[b] * p.E_BEV_CAPx_B[b]
                for k in K
                for b in B
            ),
            "BEV_last_timestep",
        )
        m.addConstr(
            (
                v.X_BEV_penalty_
                == (1 - p.z_BEV_smart_)
                * quicksum((g + k) * v.P_BEV_in_KGB[k, g, b] for g in G for k in K for b in B)
            ),
            "BEV_penalty",
        )
        m.addConstrs(
            (
                v.P_BEV_in_KGB[k, g, b]
                <= p.y_BEV_avail_KGB[k, g, b] * p.E_BEV_CAPx_B[b] * p.k_BEV_inPerCap_B[b]
                for k in K
                for g in G
                for b in B
            ),
            "BEV_max_in",
        )
        m.addConstrs(
            (
                v.P_BEV_V2X_KGB[k, g, b]
                <= p.y_BEV_avail_KGB[k, g, b]
                * p.z_BEV_V2X_
                * p.E_BEV_CAPx_B[b]
                * p.k_BEV_v2xPerCap_B[b]
                for k in K
                for g in G
                for b in B
            ),
            "BEV_max_V2X",
        )

        c.P_EL_source_KG["BEV"] = lambda k, g: v.P_BEV_V2X_KGB.sum(k, g, "*")
        c.P_EL_sink_KG["BEV"] = lambda k, g: v.P_BEV_in_KGB.sum(k, g, "*")
        c.X_TOT_penalty_["BEV"] = v.X_BEV_penalty_


@dataclass
class Elc(Component):
    """Electrolyzer"""

    dQ_CAPx: float = 0
    allow_new: bool = True

    def param_func(self, sc: Scenario):
        sc.param("P_Elc_CAPx_", data=self.dQ_CAPx, doc="Existing capacity", unit="kW_el")
        sc.param(from_db=db.eta_Elc_)
        sc.var("P_Elc_KG", doc="Consuming power", unit="kW_el")

        if sc.consider_invest:
            sc.param(from_db=db.N_Elc_)
            sc.param("z_Elc_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.c_Elc_inv_)
            sc.param(from_db=db.k_Elc_RMI_)
            sc.var("P_Elc_CAPn_", doc="New capacity", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        cap = p.P_Elc_CAPx_ + v.P_Elc_CAPn_ if sc.consider_invest else p.P_Elc_CAPx_
        m.addConstrs((v.P_Elc_KG[k, g] <= cap for k in d.K for g in d.G), "Elc_limit")
        c.P_EL_sink_KG["Elc"] = lambda k, g: v.P_Elc_KG[k, g]
        c.dH_hydrogen_source_KG["Elc"] = lambda k, g: p.eta_Elc_ * v.P_Elc_KG[k, g]

        if sc.consider_invest:
            m.addConstr((v.P_Elc_CAPn_ <= p.z_Elc_ * 1e6), "Elc_limit_new_capa")
            C_inv_ = v.P_Elc_CAPn_ * p.c_Elc_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["Elc"] = C_inv_
            c.C_TOT_invAnn_["Elc"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_Elc_)
            c.C_TOT_RMI_["Elc"] = C_inv_ * p.k_Elc_RMI_


@dataclass
class H2S(Component):
    """Hydrogen Storage"""

    H_CAPx: float = 0
    allow_new: bool = True

    def param_func(self, sc: Scenario):
        sc.param("H_H2S_CAPx_", data=self.H_CAPx, doc="Existing capacity", unit="kWh")
        # sc.param("k_H2S_ini_", data=0.9, doc="Initial and final energy filling share")
        sc.param(from_db=db.eta_H2S_ch_)
        sc.param(from_db=db.eta_H2S_dis_)
        sc.param("eta_H2S_self_", data=0.0, doc="Self-discharge per hour")
        sc.param("k_H2S_inPerCap_", data=0.7, doc="Maximum charging power per capacity")
        sc.param("k_H2S_outPerCap_", data=0.7, doc="Maximum discharging power per capacity")
        # The intra-period SOC is negative if over the typical period, more energy was withdrawn
        # than stored.
        sc.var("H_H2S_intra_KG", doc="Intra-period state of charge", unit="kWh", lb=-GRB.INFINITY)
        sc.var("H_H2S_inter_I", doc="Inter-period state of charge", unit="kWh")
        sc.var("dH_H2S_in_KG", doc="Charging power", unit="kW")
        sc.var("dH_H2S_out_KG", doc="Discharging power", unit="kW")

        if sc.consider_invest:
            sc.param(from_db=db.k_H2S_RMI_)
            sc.param(from_db=db.N_H2S_)
            sc.param("z_H2S_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.c_H2S_inv_)
            sc.var("H_H2S_CAPn_", doc="New capacity", unit="kWh")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        """For the modeling of the seasonal storage with aggregated time series, see Kotzur et al. [1].
        Note, the different formulas, since we define the time point t to be at the end of the time step t.

        [1]: https://arxiv.org/abs/1710.07593 p.9
        """
        cap = p.H_H2S_CAPx_ + v.H_H2S_CAPn_ if sc.consider_invest else p.H_H2S_CAPx_
        m.addConstrs(
            (v.dH_H2S_in_KG[k, g] <= p.k_H2S_inPerCap_ * cap for k in d.K for g in d.G),
            "H2S_limit_charging_power",
        )
        m.addConstrs(
            (v.dH_H2S_out_KG[k, g] <= p.k_H2S_outPerCap_ * cap for k in d.K for g in d.G),
            "H2S_limit_discharging_power",
        )
        m.addConstrs(
            (
                (v.H_H2S_inter_I[d.I[-1]] if i == d.I[0] else v.H_H2S_inter_I[i - 1])
                * (1 - p.eta_H2S_self_ * sc.step_width) ** p.n__stepsPerPeriod_
                + v.H_H2S_intra_KG[sc.periodsOrder[i], g]
                <= cap
                for i in d.I
                for g in d.G
            ),
            "H2S_limit_cap",
        )
        # The following step is important, since H_H2S_intra_KG can be negative:
        m.addConstrs(
            (
                (v.H_H2S_inter_I[d.I[-1]] if i == d.I[0] else v.H_H2S_inter_I[i - 1])
                * (1 - p.eta_H2S_self_ * sc.step_width) ** p.n__stepsPerPeriod_
                + v.H_H2S_intra_KG[sc.periodsOrder[i], g]
                >= 0
                for i in d.I
                for g in d.G
            ),
            "H2S_limit_cap_0",
        )
        m.addConstrs(
            (
                v.H_H2S_intra_KG[k, g]
                == (0 if g == d.G[0] else v.H_H2S_intra_KG[k, g - 1])
                * (1 - p.eta_H2S_self_ * sc.dt(k, g))
                + (v.dH_H2S_in_KG[k, g] * p.eta_H2S_ch_ - v.dH_H2S_out_KG[k, g] / p.eta_H2S_dis_)
                * sc.dt(k, g)
                for k in d.K
                for g in d.G
            ),
            "H2S_balance_intra-period",
        )

        # The inter-period state of charge (SOC) of a period i is:
        #   the inter-period SOC of the previous period (i-1)
        #   minus the self discharge of the whole period
        #   plus the balance of the last intra-period SOC of respective typical period.
        m.addConstrs(
            (
                v.H_H2S_inter_I[i]
                == (v.H_H2S_inter_I[d.I[-1]] if i == d.I[0] else v.H_H2S_inter_I[i - 1])
                * (1 - p.eta_H2S_self_ * sc.step_width) ** p.n__stepsPerPeriod_
                + v.H_H2S_intra_KG[sc.periodsOrder[i], d.G[-1]]
                for i in d.I
            ),
            "H2S_balance_inter-period",
        )

        c.dH_hydrogen_source_KG["H2S"] = lambda k, g: v.dH_H2S_out_KG[k, g]
        c.dH_hydrogen_sink_KG["H2S"] = lambda k, g: v.dH_H2S_in_KG[k, g]
        if sc.consider_invest:
            m.addConstr((v.H_H2S_CAPn_ <= p.z_H2S_ * 1e6), "H2S_limit_new_capa")
            C_inv_ = v.H_H2S_CAPn_ * p.c_H2S_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["H2S"] = C_inv_
            c.C_TOT_invAnn_["H2S"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_H2S_)
            c.C_TOT_RMI_["H2S"] = C_inv_ * p.k_H2S_RMI_

    def postprocess_func(self, sc: Scenario):
        d, r = sc.dims, sc.res
        name = "H_H2S_T"
        # `ser = r.H_H2S_inter_T + r.H_H2S_intra_T` IS NOT valid, due to time series aggregation!
        ser = r.H_H2S_inter_I[d.I[-1]] + (r.dH_H2S_in_T - r.dH_H2S_out_T).cumsum() * sc.step_width
        ser = ser.rename(name)
        setattr(r, name, ser)


@dataclass
class FC(Component):
    """Fuel Cell

    Hydrogen balance:
    FC   <-- |
    Elc  --> |
    H2S  <-> |
    """

    P_CAPx: float = 0
    allow_new: bool = True
    H_level_target: str = "70/40"

    def param_func(self, sc: Scenario):
        sc.param("P_FC_CAPx_", data=self.P_CAPx, doc="Existing capacity", unit="kW_el")
        sc.param(from_db=db.eta_FC_el_)
        sc.param(from_db=db.eta_FC_th_)
        sc.var("dQ_FC_KG", doc="Producing heat flow", unit="kW_th")
        sc.var("dH_FC_KG", doc="Consumed hydrogen flow", unit="kW")
        sc.var("P_FC_FI_KG", doc="Feed-in", unit="kW_el")
        sc.var("P_FC_OC_KG", doc="Own consumption", unit="kW_el")
        sc.var("P_FC_KG", doc="Producing electrical power", unit="kW_el")

        if sc.consider_invest:
            sc.param("z_FC_", data=int(self.allow_new), doc="If new capacity is allowed")
            sc.param(from_db=db.c_FC_inv_)
            sc.param(from_db=db.k_FC_RMI_)
            sc.param(from_db=db.N_FC_)
            sc.var("P_FC_CAPn_", doc="New capacity", unit="kW_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (v.P_FC_KG[k, g] == p.eta_FC_el_ * v.dH_FC_KG[k, g] for k in d.K for g in d.G),
            "FC_elec_balance",
        )
        m.addConstrs(
            (v.dQ_FC_KG[k, g] == p.eta_FC_th_ * v.dH_FC_KG[k, g] for k in d.K for g in d.G),
            "FC_heat_balance",
        )
        cap = p.P_FC_CAPx_ + v.P_FC_CAPn_ if sc.consider_invest else p.P_FC_CAPx_
        m.addConstrs((v.P_FC_KG[k, g] <= cap for k in d.K for g in d.G), "FC_limit_elecPower")
        m.addConstrs(
            (v.P_FC_KG[k, g] == v.P_FC_FI_KG[k, g] + v.P_FC_OC_KG[k, g] for k in d.K for g in d.G),
            "FC_feedIn_vs_ownConsumption",
        )
        c.P_EL_source_KG["FC"] = lambda k, g: v.P_FC_FI_KG[k, g] + v.P_FC_OC_KG[k, g]
        c.dQ_heating_source_KGH["FC"] = (
            lambda k, g, h: v.dQ_FC_KG[k, g] if h == self.H_level_target else 0
        )
        c.dH_hydrogen_sink_KG["FC"] = lambda k, g: v.dH_FC_KG[k, g]
        c.P_EG_sell_KG["FC"] = lambda k, g: v.P_FC_FI_KG[k, g]

        if sc.consider_invest:
            m.addConstr((v.P_FC_CAPn_ <= p.z_FC_ * 1e6), "FC_limit_new_capa")
            C_inv_ = v.P_FC_CAPn_ * p.c_FC_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_inv_["FC"] = C_inv_
            c.C_TOT_invAnn_["FC"] = C_inv_ * get_annuity_factor(r=p.k__r_, N=p.N_FC_)
            c.C_TOT_RMI_["FC"] = C_inv_ * p.k_FC_RMI_


order_restrictions = [
    ("cDem", {}),
    ("hDem", {}),
    ("eDem", {}),
    ("EG", {"PV", "CHP", "WT", "FC"}),  # EG collects P_EG_sell_KG
    ("Fuel", {"HOB", "CHP"}),  # Fuel collects F_fuel_F
    ("FC", {}),
    ("H2S", {}),
    ("Elc", {}),
    ("BES", {}),
    ("BEV", {}),
    ("PV", {}),
    ("DAC", {}),
    ("WT", {}),
    ("P2H", {}),
    ("CHP", {}),
    ("HOB", {}),
    ("HD", {}),
    ("HP", {"cDem", "hDem"}),  # HP calculates COP based on thermal demand temperatures
    ("TES", {"cDem", "hDem"}),  # TESs can be defined for every thermal demand temperature level
]
order_restrictions.append(("Main", [x[0] for x in order_restrictions]))

set_component_order_by_order_restrictions(order_restrictions=order_restrictions, classes=globals())
