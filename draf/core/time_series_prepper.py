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

    def k__PartYearComp_(self, name: str = "k__PartYearComp_") -> float:
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
    def ce_EG_T(self, name: str = "ce_EG_T", method: str = "XEF_PP", **kwargs) -> pd.Series:
        """Add dynamic carbon emission factors."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="kgCO2eq/kWh_el",
            doc=f"Carbon emission factors (via elmada using year, freq, country, and CEF-method)",
            data=sc.match_dtindex(
                get_emissions(
                    year=sc.year, freq=sc.freq, country=sc.country, method=method, **kwargs
                )
                * hp.conv("g", "kg", 1e-3)
            ),
        )

    def c_EG_RTP_T(self, name: str = "c_EG_RTP_T", method: str = "hist_EP", **kwargs) -> pd.Series:
        """Add Real-time-prices-tariffs."""
        sc = self.sc
        return self.sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Day-ahead-market-prices (via elmada using year, freq, and country)",
            data=sc.match_dtindex(
                get_prices(year=sc.year, freq=sc.freq, method=method, country=sc.country, **kwargs)
                / 1e3
            ),
        )

    def c_EG_PP_T(self, name: str = "c_EG_PP_T", method: str = "PP") -> pd.Series:
        """Add marginal costs from PP-method. Only for Germany."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=sc.match_dtindex(
                get_prices(year=sc.year, freq=sc.freq, country=sc.country, method=method)
            ),
        )

    def c_EG_PWL_T(self, name: str = "c_EG_PWL_T", method: str = "PWL", **kwargs) -> pd.Series:
        """Add marginal costs from PWL-method."""
        sc = self.sc
        return sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Marginal Costs {sc.year}, {sc.freq}, {sc.country}",
            data=sc.match_dtindex(
                get_prices(year=sc.year, freq=sc.freq, country=sc.country, method=method, **kwargs)
            ),
        )

    def c_EG_TOU_T(
        self,
        name: str = "c_EG_TOU_T",
        prices: Optional[Tuple[float, float]] = None,
        prov: str = "BW",
    ) -> pd.Series:
        """A Time-of-Use tariff with two prices.
        If no prices are given the according RTP tariff is taken as basis.
        """
        holis = getattr(holidays, self.sc.country)(subdiv=prov)
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
                low_price = self.sc.params.c_EG_RTP_T[isLowTime_T].mean()
                high_price = self.sc.params.c_EG_RTP_T[isHighTime_T].mean()
            except AttributeError as err:
                logger.error(
                    "Mean price for TOU tariff cannot be inferred"
                    f" from RTP, since there is no RTP. {err}"
                )

        else:
            if isinstance(prices, Tuple) and len(prices) == 2:
                low_price = min(prices)
                high_price = max(prices)

        return self.sc.param(
            name=name,
            unit="€/kWh_el",
            doc=f"Time-Of-Use-tariff (calculated from Real-time-price)",
            data=low_price * isLowTime_T + high_price * isHighTime_T,
        )

    def c_EG_FLAT_T(self, price: Optional[float] = None, name: str = "c_EG_FLAT_T"):
        """Add a flat electricity tariff.

        If no price is given the according RTP tariff is taken as basis.
        """
        if price is None:
            try:
                price = self.sc.params.c_EG_RTP_T.mean()
            except AttributeError as err:
                logger.error(
                    "Mean price for FLAT tariff cannot be inferred"
                    f" from RTP, since there is no RTP. {err}"
                )

        unit = "€/kWh_el"
        return self.sc.param(
            name=name,
            unit=unit,
            doc=f"Flat-electricity tariff (calculated from Real-time-price)",
            fill=price,
        )

    @hp.copy_doc(prep.get_el_SLP)
    def P_eDem_T(
        self,
        name: str = "P_eDem_T",
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
            data=sc.match_dtindex(
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

    @hp.copy_doc(prep.get_heating_demand)
    def dQ_hDem_T(
        self,
        name: str = "dQ_hDem_T",
        annual_energy: float = 1e6,
        target_temp: float = 22.0,
        threshold_temp: float = 15.0,
    ) -> pd.Series:
        """Create and add a heating demand time series using weather data nearby."""

        sc = self.sc

        ser_amb_temp = prep.get_air_temp(coords=sc.coords, year=sc.year)

        return sc.param(
            name=name,
            unit="kW_th",
            doc=f"Heating demand derived from ambient temperature near coords.",
            data=sc.match_dtindex(
                prep.get_heating_demand(
                    year=sc.year,
                    freq=sc.freq,
                    ser_amb_temp=ser_amb_temp,
                    annual_energy=annual_energy,
                    target_temp=target_temp,
                    threshold_temp=threshold_temp,
                )
            ),
        )

    @hp.copy_doc(prep.get_cooling_demand)
    def dQ_cDem_T(
        self,
        name: str = "dQ_cDem_T",
        annual_energy: float = 1e6,
        target_temp: float = 22.0,
        threshold_temp: float = 22.0,
    ) -> pd.Series:
        """Create and add a cooling demand time series using weather data nearby."""
        sc = self.sc

        return sc.param(
            name=name,
            unit="kW_th",
            doc=f"Cooling demand derived from ambient temperature near coords.",
            data=sc.match_dtindex(
                prep.get_cooling_demand(
                    year=sc.year,
                    freq=sc.freq,
                    coords=sc.coords,
                    annual_energy=annual_energy,
                    target_temp=target_temp,
                    threshold_temp=threshold_temp,
                )
            ),
        )

    def P_PV_profile_T(
        self,
        name: str = "P_PV_profile_T",
        use_coords: bool = True,
        overwrite_coords: Optional[Tuple] = None,
        **gsee_kw,
    ) -> pd.Series:
        """Add a photovoltaic profile.

        Args:
            use_coords: For Germany only: If the `coords` of the CaseStudy should be used to
                calculate the PV profile via `gsee`. In that case, the weather data from the
                nearest available weather station is used.
            overwrite_coords: Coordinates that are taken instead of the case study coordinates.
            gsee_kw: Keywords used in the `gsee.pv.run_model` (https://gsee.readthedocs.io)
                function.
        """
        sc = self.sc

        if use_coords:
            if sc.coords is not None:
                coords = sc.coords
            if overwrite_coords is not None:
                coords = overwrite_coords
            logger.info(f"{coords} coordinates used for PV calculation.")
            assert coords is not None, "No coordinates given, but `use_coords=True`."
            ser = prep.get_pv_power(year=sc.year, coords=coords, **gsee_kw).reset_index(drop=True)
        else:
            logger.warning(
                "No coords given or usage not wanted. Year-independant backup PV profile is used."
            )
            ser = prep.get_backup_PV_profile()
            import calendar

            if calendar.isleap(self.sc.year):
                ser = pd.Series(np.concatenate([ser.values, ser[-24:].values]))

        return sc.param(
            name=name,
            unit="kW_el/kW_peak",
            doc="Produced PV-power for 1 kW_peak",
            data=sc.match_dtindex(ser, resample=True),
        )

    def c_EG_addon_(
        self,
        name: str = "c_EG_addon_",
        AbLa_surcharge=0.00003,
        Concession_fee=0.0011,
        EEG_surcharge=0,  # (German Renewable Energies Act levy) no longer due since 2022-07-01
        Electricity_tax=0.01537,
        KWK_surcharge=0.0038,
        Network_cost=0,  # paid per kW peak
        NEV_surcharge=0.0027,
        Offshore_surcharge=0.00419,
        Sales=0.01537,
    ) -> float:
        """Add electricity price components other than wholesale prices.
        Defaults for German Industry [1].

        [1]: https://www.bdew.de/media/documents/221208_BDEW-Strompreisanalyse_Dez2022_08.12.2022_korr_vx5gByn.pdf page 34
        """

        price_components = [
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
            doc="Electricity taxes and levies",
            data=sum(price_components),
        )

    def T__amb_T(self, name: str = "T__amb_T") -> pd.Series:
        """Uses coordinates to prepare ambient air temperature time series in °C."""
        sc = self.sc
        assert isinstance(sc.coords, tuple)
        ser = prep.get_air_temp(coords=sc.coords, year=sc.year, with_dt=False)
        return sc.param(
            name=name,
            unit="°C",
            doc=f"Ambient temperature",
            data=sc.match_dtindex(sc.resample(ser)),
        )
