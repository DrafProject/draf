import logging
from typing import Any, List, Optional, Tuple

import holidays
import numpy as np
import pandas as pd
from elmada import get_emissions, get_prices

import draf.helper as hp
from draf import prep
from draf.prep.demand import SLP_PROFILES

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class TimeSeriesPrepper:
    """This class holds convenience functions for paremeter preparation and hands the scenario
    object on them.
    """

    from draf.prep import param_funcs as funcs

    def __init__(self, sc):
        self.sc = sc

    def __getstate__(self):
        """For serialization with pickle."""
        return None

    def k_comp_(self, name: str = "k_comp_") -> float:
        """Add cost weighting factor to compensate part year analysis."""
        sc = self.sc
        return self.sc.param(
            name=name,
            unit="",
            doc="Weighting factor to compensate part year analysis",
            data=len(sc.dtindex) / len(sc.dtindex_custom),
        )

    def k__dT_(self, name: str = "k__dT_"):
        return self.sc.param(name=name, unit="h", doc="Time steps width", data=self.sc.step_width)

    @hp.copy_doc(get_emissions, start="Args:")
    def ce_GRID_T(self, name: str = "ce_GRID_T", method: str = "XEF_PP", **kwargs) -> pd.Series:
        """Add dynamic carbon emission factors."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="kgCO2eq/kWh_el",
            doc=f"{method} for {sc.year}, {sc.freq}, {sc.country}",
            data=sc.trim_to_datetimeindex(
                get_emissions(
                    year=sc.year,
                    freq=sc.freq,
                    country=sc.country,
                    method=method,
                    **kwargs,
                )
                / 1e3
            ),
        )

    def c_GRID_RTP_T(
        self, name: str = "c_GRID_RTP_T", method: str = "hist_EP", **kwargs
    ) -> pd.Series:
        """Add Real-time-prices-tariffs."""
        sc = self.sc
        return self.sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Day-ahead-market-prices {sc.year}, {sc.freq}, {sc.country}",
            data=sc.trim_to_datetimeindex(
                get_prices(year=sc.year, freq=sc.freq, method=method, country=sc.country, **kwargs)
                / 1000
            ),
        )

    def c_GRID_PP_T(self, name: str = "c_GRID_PP_T", method: str = "PP") -> pd.Series:
        """Add marginal costs from PP-method. Only for Germany."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=sc.trim_to_datetimeindex(
                get_prices(year=sc.year, freq=sc.freq, country=sc.country, method=method)
            ),
        )

    def c_GRID_PWL_T(self, name: str = "c_GRID_PWL_T", method: str = "PWL", **kwargs) -> pd.Series:
        """Add marginal costs from PWL-method."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=sc.trim_to_datetimeindex(
                get_prices(year=sc.year, freq=sc.freq, country=sc.country, method=method, **kwargs)
            ),
        )

    def c_GRID_TOU_T(
        self,
        name: str = "c_GRID_TOU_T",
        prices: Optional[Tuple[float, float]] = None,
        prov: str = "BW",
    ) -> pd.Series:
        """A Time-of-Use tariff with two prices.
        If no prices are given the according RTP tariff is taken as basis.
        """
        sc = self.sc

        holis = getattr(holidays, sc.country)(prov=prov)

        isLowTime_T = np.array(
            [
                True
                if ((x.dayofweek >= 5) or not (8 <= x.hour < 20) or (x.date() in holis))
                else False
                for x in self.sc.dtindex_custom
            ]
        )
        isHighTime_T = np.invert(isLowTime_T)

        if prices is None:
            try:
                low_price = self.sc.params.c_GRID_RTP_T[isLowTime_T].mean()
                high_price = self.sc.params.c_GRID_RTP_T[isHighTime_T].mean()
            except AttributeError as err:
                logger.error(
                    f"Mean price for TOU tariff cannot be inferred"
                    f" from RTP, since there is no RTP. {err}"
                )

        else:
            if isinstance(prices, Tuple) and len(prices) == 2:
                low_price = min(prices)
                high_price = max(prices)

        return sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Time-Of-Use-tariff with the prices {low_price:.3f}€ and {high_price:.3f}€",
            data=low_price * isLowTime_T + high_price * isHighTime_T,
        )

    def c_GRID_FLAT_T(
        self, price: Optional[float] = None, name: str = "c_GRID_FLAT_T", doc_addon: str = ""
    ):
        """Add a flat electricity tariff.

        If no price is given the according RTP tariff is taken as basis.
        """
        if price is None:
            try:
                price = self.sc.params.c_GRID_RTP_T.mean()
            except AttributeError as err:
                logger.error(
                    f"Mean price for FLAT tariff cannot be inferred"
                    f" from RTP, since there is no RTP. {err}"
                )

        unit = "€/kWh_el"
        return self.sc.param(
            name=name,
            unit=unit,
            doc=f"Flat-electricity tariff ({price:.4f} {unit}). {doc_addon}",
            fill=price,
        )

    @hp.copy_doc(prep.get_el_SLP)
    def P_dem_T(
        self,
        name: str = "P_dem_T",
        profile: str = "G1",
        peak_load: Optional[float] = None,
        annual_energy: Optional[float] = None,
        offset: float = 0,
        province: Optional[str] = None,
    ) -> pd.Series:
        """Add an electricity demand"""
        sc = self.sc

        return sc.param(
            name=name,
            unit="kW_el",
            doc=f"Electricity demand from standard load profile {profile}: {SLP_PROFILES[profile]}",
            data=sc.trim_to_datetimeindex(
                prep.get_el_SLP(
                    year=sc.year,
                    freq=sc.freq,
                    profile=profile,
                    peak_load=peak_load,
                    annual_energy=annual_energy,
                    offset=offset,
                    country=sc.country,
                    province=province,
                )
            ),
        )

    def H_dem_H_TH(self) -> pd.Series:
        data = pd.Series(0, pd.MultiIndex.from_product([self.sc.dims.T, self.sc.dims.H]))
        return self.sc.param(name="H_dem_H_TH", data=data, doc="Heating demand", unit="kW_th")

    def H_dem_C_TN(self) -> pd.Series:
        data = pd.Series(0, pd.MultiIndex.from_product([self.sc.dims.T, self.sc.dims.N]))
        return self.sc.param(name="H_dem_C_TN", data=data, doc="Cooling demand", unit="kW_th")

    @hp.copy_doc(prep.get_heating_demand)
    def H_dem_H_T(
        self,
        name: str = "H_dem_H_T",
        annual_energy: float = 1e6,
        target_temp: float = 22.0,
        threshold_temp: float = 15.0,
    ) -> pd.Series:
        """Add a heat demand based on the `target_temp`, `threshold_temp`, `annual_energy`."""
        sc = self.sc

        ser_amb_temp = prep.get_air_temp(coords=sc.coords, year=sc.year)

        return sc.param(
            name=name,
            unit="kWh_th",
            doc=f"Heating demand derived from ambient temperature near {sc.coords}.",
            data=sc.trim_to_datetimeindex(
                prep.get_heating_demand(
                    ser_amb_temp=ser_amb_temp,
                    annual_energy=annual_energy,
                    target_temp=target_temp,
                    threshold_temp=threshold_temp,
                )
            ),
        )

    @hp.copy_doc(prep.get_cooling_demand)
    def H_dem_C_T(
        self,
        name: str = "H_dem_C_T",
        annual_energy: float = 1e6,
        target_temp: float = 22.0,
        threshold_temp: float = 22.0,
    ) -> pd.Series:
        """Add a heat demand based on the `target_temp`, `threshold_temp`, `annual_energy`."""
        sc = self.sc

        ser_amb_temp = prep.get_air_temp(coords=sc.coords, year=sc.year)

        return sc.param(
            name=name,
            unit="kW_th",
            doc=f"Cooling demand derived from ambient temperature near {sc.coords}.",
            data=sc.trim_to_datetimeindex(
                prep.get_cooling_demand(
                    ser_amb_temp=ser_amb_temp,
                    annual_energy=annual_energy,
                    target_temp=target_temp,
                    threshold_temp=threshold_temp,
                )
            ),
        )

    def P_PV_profile_T(
        self, name: str = "P_PV_profile_T", use_coords: bool = True, **gsee_kw
    ) -> pd.Series:
        """Add a photovoltaic profile.

        For Germany only: If `coords` are given as within the CaseStudy
        a `gsee` calculation is conducted with weather data from the nearest available weather
        station.
        """
        sc = self.sc

        if sc.coords is not None and use_coords:
            logger.info(f"{sc.coords} coordinates used for PV calculation.")
            ser = prep.get_pv_power(year=sc.year, coords=sc.coords, **gsee_kw).reset_index(
                drop=True
            )
        else:
            logger.warning(
                "No coords given or usage not wanted. Year-independant backup PV profile is used."
            )
            ser = prep.get_backup_PV_profile()
            import calendar

            if calendar.isleap(self.sc.year):
                ser = pd.Series(np.concatenate([ser.values, ser[-24:].values]))

        ser = hp.resample(ser, year=sc.year, start_freq=hp.estimate_freq(ser), target_freq=sc.freq)

        return sc.param(
            name=name,
            unit="kW_el/kW_peak",
            doc=f"Produced PV-power for 1 kW_peak",
            data=sc.trim_to_datetimeindex(ser),
        )

    def c_GRID_addon_T(
        self,
        name: str = "c_GRID_addon_T",
        AbLa_surcharge=0.00006,
        Concession_fee=0.0011,
        EEG_surcharge=0.0688,
        Electricity_tax=0.01537,
        KWK_surcharge=0.0029,
        Network_cost=0.025,
        NEV_surcharge=0.0025,
        Offshore_surcharge=-0.00002,
        Sales=0.01537,
    ) -> pd.Series:
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

        return self.sc.param(
            name=name,
            unit="€/kWh_el",
            doc="Add-on electricity price component",
            data=pd.Series(sum(components), self.sc.dims.T),
        )
