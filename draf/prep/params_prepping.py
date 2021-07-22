import logging
from typing import Any, List, Optional

import draf.helper as hp
import holidays
import numpy as np
import pandas as pd
from draf.io import get_ambient_temp, get_PV_profile, get_SLP, get_thermal_demand
from elmada import get_emissions, get_prices

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class Prepper:
    """This class holds convenience functions for paremeter preparation and hands the scenario
     object on them.
    """

    def __init__(self, sc):
        self.sc = sc

    def __getstate__(self):
        """For serialization with pickle."""
        return None

    def add_n_comp_(self, name="n_comp_"):
        """Add cost weighting factor to compensate part year analysis."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="",
            doc="Weighting factor to compensate part year analysis",
            data=len(sc.dtindex) / len(sc.dtindex_custom),
        )

    @hp.copy_doc(get_emissions, start="Args:")
    def add_ce_GRID_T(self, name="ce_GRID_T", method="XEF_PP", **kwargs):
        """Add dynamic carbon emission factors."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="kgCO2eq/kWh_el",
            doc=f"{method} for {sc.year}, {sc.freq}, {sc.country}",
            data=get_emissions(
                year=sc.year, freq=sc.freq, country=sc.country, method=method, **kwargs,
            )[sc._t1 : sc._t2 + 1]
            / 1e3,
        )

    def add_c_GRID_RTP_T(self, name="c_GRID_RTP_T", method="hist_EP", **kwargs):
        """Add Real-time-prices-tariffs."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="€/kWh_el",
            doc=f"Day-ahead-market-prices {sc.year}, {sc.freq}, {sc.country}",
            data=get_prices(
                year=sc.year, freq=sc.freq, method=method, country=sc.country, **kwargs
            )[sc._t1 : sc._t2 + 1]
            / 1000,
        )

    def add_c_GRID_PP_T(self, name="c_GRID_PP_T", method="PP"):
        """Add marginal costs from PP-method. Only for Germany."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=get_prices(year=sc.year, freq=sc.freq, country=sc.country, method=method)[
                sc._t1 : sc._t2 + 1
            ],
        )

    def add_c_GRID_PWL_T(self, name="c_GRID_PWL_T", method="PWL", **kwargs):
        """Add marginal costs from PWL-method."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=get_prices(
                year=sc.year, freq=sc.freq, country=sc.country, method=method, **kwargs
            )[sc._t1 : sc._t2 + 1],
        )

    def add_c_GRID_TOU_T(
        self, name: str = "c_GRID_TOU_T", prices: Optional[List[float]] = None, prov: str = "BW"
    ):
        """A Time-of-Use tariff with two prices.
        If no prices are given the according RTP tariff is taken as reference.
        """
        sc = self.sc

        holis = getattr(holidays, sc.country)(prov=prov)

        # calculate high/low times:
        _y_lt = np.array(
            [
                True
                if ((x.dayofweek >= 5) or not (8 <= x.hour < 20) or (x.date() in holis))
                else False
                for x in self.sc.dtindex_custom
            ]
        )
        _y_ht = np.invert(_y_lt)

        if prices is None:
            try:
                _lt = self.sc.params.c_GRID_RTP_T[_y_lt].mean()
                _ht = self.sc.params.c_GRID_RTP_T[_y_ht].mean()
            except AttributeError as err:
                logger.error(
                    f"Mean prices for TOU tariff cannot be inferred "
                    f"from RTP, since there is no RTP. {err}"
                )

        else:
            if isinstance(prices, list) and len(prices) == 2:
                _lt = min(prices)
                _ht = max(prices)

        return self.sc.add_par(
            name=name,
            unit="€/kWh_el",
            doc=f"Time-Of-Use-tariff with the prices {_lt:.3f}€ and {_ht:.3f}€",
            data=_lt * _y_lt + _ht * _y_ht,
        )

    def add_c_GRID_FLAT_T(
        self, price: Optional[float] = None, name="c_GRID_FLAT_T", doc_addon: str = ""
    ):
        if price is None:
            try:
                price = self.sc.params.c_GRID_RTP_T.mean()
            except AttributeError as err:
                logger.error(
                    f"Mean price for FLAT tariff cannot be inferred"
                    f" from RTP, since there is no RTP. {err}"
                )

        unit = "€/kWh_el"
        return self.sc.add_par(
            name=name,
            unit=unit,
            doc=f"Flat-electricity tariff ({price:.4f} {unit}). {doc_addon}",
            fill=price,
        )

    @hp.copy_doc(get_SLP)
    def add_E_dem_T(
        self,
        name="E_dem_T",
        profile="G1",
        peak_load: Optional[float] = None,
        annual_energy: Optional[float] = None,
        offset: float = 0,
        province: Optional[str] = None,
    ):
        """Add an electricity demand"""
        sc = self.sc

        return self.sc.add_par(
            name=name,
            unit="kWh_el",
            doc=f"Electricity demand from standard load profile {profile}",
            data=get_SLP(
                year=sc.year,
                freq=sc.freq,
                profile=profile,
                peak_load=peak_load,
                annual_energy=annual_energy,
                offset=offset,
                country=sc.country,
                province=province,
            )[sc._t1 : sc._t2 + 1],
        )

    @hp.copy_doc(get_thermal_demand)
    def add_H_dem_T(
        self,
        name="H_dem_T",
        annual_energy: float = 1.0,
        target_temp: float = 22.0,
        threshold_temp: float = 15.0,
        location="Rheinstetten",
    ):
        """Add a heat demand based on the `target_temp`, `threshold_temp`, `annual_energy`."""
        sc = self.sc

        ser_amb_temp = get_ambient_temp(year=sc.year, freq=sc.freq, location=location)

        return self.sc.add_par(
            name=name,
            unit="kWh_th",
            doc=f"Heat demand derived from ambient temperatur in {location}",
            data=get_thermal_demand(
                ser_amb_temp=ser_amb_temp,
                annual_energy=annual_energy,
                target_temp=target_temp,
                threshold_temp=threshold_temp,
            )[sc._t1 : sc._t2 + 1],
        )

    def add_E_PV_profile_T(self, name="E_PV_profile_T"):
        """Add a photovoltaic profile."""
        sc = self.sc
        return self.sc.add_par(
            name=name,
            unit="kW_el/kW_peak",
            doc=f"Produced PV-power for 1 kW_peak",
            data=get_PV_profile()[sc._t1 : sc._t2 + 1],
        )

    def add_c_GRID_addon_T(
        self,
        name="c_GRID_addon_T",
        AbLa_surcharge=0.00006,
        Concession_fee=0.0011,
        EEG_surcharge=0.0688,
        Electricity_tax=0.01537,
        KWK_surcharge=0.0029,
        Network_cost=0.025,
        NEV_surcharge=0.0025,
        Offshore_surcharge=-0.00002,
        Sales=0.01537,
    ):
        """Add electricity price components other than wholesale prices defaults for Industry for
        2017.

        Source for defaults: https://www.bdew.de/media/documents/190723_BDEW-Strompreisanalyse_Juli-2019.pdf page 25
        """

        components = [
            AbLa_surcharge,
            Concession_fee,
            EEG_surcharge,
            Electricity_tax,
            KWK_surcharge,
            Network_cost,
            NEV_surcharge,
            Offshore_surcharge,
            Sales,
        ]

        return self.sc.add_par(
            name=name,
            unit="€/kWh_el",
            doc="Add-on electricity price component",
            data=pd.Series(sum(components), self.sc.dims.T),
        )
