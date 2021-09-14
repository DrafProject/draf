import logging
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from elmada.helper import read
from gsee import pv as gsee_pv

from draf import paths
from draf.prep.weather import get_data_for_gsee, get_nearest_stations

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


def fit_gsee_model_to_real(
    real: pd.Series, gsee_params: Dict, capa: float, get_whole_year_if_sliced: bool = True
) -> Tuple[pd.Series, float]:
    """Runs the gsee model and fits the resulting power time series according
     to the annual energy sum.

    Args:
        real: Hourly historic PV power.
        gsee_params: Year, tilt, azim, system_loss.
        capa: PV Capacity in kW_peak.
        get_whole_year_if_sliced: If True, models the whole year even if only part-year
            data were given.

    Returns:
        pd.Series: Resulting PV-power time series.
        float: Additional system losses not caused by panel and inverter (fraction).
    """
    slicer = _get_slicer(real)
    e_gsee = get_pv_power(**gsee_params)
    gsee_energy = capa * e_gsee[slicer].sum()
    real_energy = real[slicer].sum()
    system_loss = (gsee_energy - real_energy) / gsee_energy
    ser = capa * e_gsee * (1 - system_loss)
    result_slicer = slice(None) if get_whole_year_if_sliced else slicer
    return ser[result_slicer], system_loss


def _get_slicer(real: pd.Series) -> slice:
    ser = real[real.notnull()]
    start = ser.index[0]
    end = ser.index[-1]
    is_whole_year = start.day == 1 and start.month == 1 and end.day == 31 and end.month == 12

    if is_whole_year:
        return slice(None)
    else:
        logger.warning(f"Part-year data were given: {start} - {end}.")
        return slice(start, end)


@lru_cache(maxsize=5)
def get_pv_power(
    year: int,
    coords: Tuple[float, float],
    tilt: float = 0,
    azim: float = 180,
    capacity: float = 1.0,
    tracking: int = 0,
    system_loss: float = 0.0,
    **gsee_kw,
):
    """Returns electrical PV power using the gsee.pv model with weather data from
     the nearest DWD weather station.

    Args:
        coords: Latitude and longitude.
        tilt: Tilt angle (degrees).
        azim: Azimuth angle (degrees, 180 = towards equator).
        tracking: Tracking (0: none, 1: 1-axis, 2: 2-axis).
        capacity : Installed capacity in W.
        system_loss: Total system power losses (fraction).
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    df = get_nearestStationData_for_gsee(year=year, coords=coords)
    return gsee_pv.run_model(
        data=df,
        coords=coords,
        tilt=tilt,
        azim=azim,
        tracking=tracking,
        capacity=capacity,
        system_loss=system_loss,
        **gsee_kw,
    )


def get_nearestStationData_for_gsee(year: int, coords: Tuple[float, float]):
    meta = get_nearest_stations(coords=coords, year=year)
    logger.info(f"Used stations:\n{meta.to_string()}")
    return get_data_for_gsee(
        stations_id_air=meta.loc["Stations_id", "air_temperature"],
        stations_id_solar=meta.loc["Stations_id", "solar"],
        year=year,
    )


def get_backup_PV_profile() -> pd.Series:
    """Get a 60min backup PV profile for 1 kWh_peak for a unspecific non-leapyear."""
    return read(paths.DATA_DIR / "pv/backup/pv_el.csv")
