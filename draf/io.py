"""Interface functions between draf core and its modules prep and data."""

import datetime
import logging
import warnings
from io import StringIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import holidays
import pandas as pd
from elmada.helper import read, write

from draf import helper as hp
from draf import paths

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


def get_SLP(
    year: int = 2019,
    freq: str = "15min",
    profile: str = "G1",
    peak_load: Optional[float] = None,
    annual_energy: Optional[float] = None,
    offset: float = 0,
    country: str = "DE",
    province: Optional[str] = None,
) -> pd.Series:
    """Returns a synthetic time series based on standard load profiles.

    Args:
        year: Year
        freq: Desired frequency of the resulting time series. Choose between '60min', '15min'.
        profile: Chosen profile. Choose between:
            H0:  Household general
            G0:  Business in general
            G1:  Business on weekdays 08:00 - 18:00
            G2:  Business with heavy evening consumption
            G3:  Business continuous
            G4:  shop/hairdresser
            G5:  Bakeries with bakery
            G6:  Weekend operation
            L0:  farms
            L1:  farms with milk and livestock
            L2:  other farms
        peak_load: Maximum load of the year. If given, the time series is scaled so that the
            maximum value matches it.
        annual_energy: Yearly energy value in kWh_el. If given, the time series is scaled so
            that the yearly total matches the `annual_energy`.
        offset: Value which is added to every time step and also considered then scaling to
            `peak_load` and `annual_energy`.
        country: Used to determine the public holidays.
            E.g. if `country`='DE' and `province`=None public holidays for North Rhine-Westphalia
            are taken: Neujahrstag, Karfreitag, Ostersonntag, Ostermontag, Maifeiertag,
            Christi Himmelfahrt, Pfingstsonntag, Pfingstmontag, Fronleichnam,
            Tag der Deutschen Einheit, Allerheiligen, 1. Weihnachtstag, 2. Weihnachtstag
        province: Further specifies the public holidays more specifically.

    Season Types:
        Summer:       15.05. - 14.09.
        Winter:       01.11. - 20.03.
        Transitional: 21.03. - 14.05. and 15.09. - 31.10.

    Public holidays (dependent on country and province or state) receive the Sunday profile.
    """
    warnings.filterwarnings("ignore", message="indexing past lexsort depth")

    profiles = {
        "H0": "Household general",
        "G0": "Business in general",
        "G1": "Business on weekdays 08:00 - 18:00",
        "G2": "Business with heavy evening consumption",
        "G3": "Business continuous",
        "G4": "shop/hairdresser",
        "G5": "Bakeries with bakery",
        "G6": "Weekend operation",
        "L0": "farms",
        "L1": "farms with milk and livestock",
        "L2": "other farms",
    }

    assert profile in profiles

    if offset > 0 and peak_load is not None:
        assert offset < peak_load

    fp = paths.DATA / f"demand/electricity/SLP/Lastprofil_{profile}.xls"
    df1 = pd.read_excel(io=fp, skiprows=[1, 2, 3, 4, 5, 6, 7, 9], header=[1], usecols=range(0, 5))
    df2 = pd.read_excel(io=fp, skiprows=[1, 2, 3, 4, 5, 6, 7, 9], header=[1], usecols=range(6, 11))
    df3 = pd.read_excel(io=fp, skiprows=[1, 2, 3, 4, 5, 6, 7, 9], header=[1], usecols=range(12, 17))

    df2.columns = df1.columns
    df3.columns = df1.columns

    df = df1.append(df2).append(df3)

    df = df.set_index(["Jahreszeit", "Tagestyp", "Zeit von"])
    df = df.drop(columns="Zeit bis")

    def _get_season(date):
        is_summer = (datetime.datetime(date.year, 5, 15) <= date) and (
            date <= datetime.datetime(date.year, 9, 14)
        )
        is_winter = (date >= datetime.datetime(date.year, 11, 1)) or (
            date <= datetime.datetime(date.year, 3, 20)
        )
        if is_summer:
            return "Sommer"
        elif is_winter:
            return "Winter"
        else:
            return "Übergangszeit"

    holiday_obj = getattr(holidays, country)(prov=province)

    def _get_day_type(date):
        is_holiday = date in holiday_obj
        is_sunday = date.dayofweek == 6
        is_saturday = date.dayofweek == 5
        if is_sunday or is_holiday:
            return "Sonntag"
        elif is_saturday:
            return "Samstag"
        else:
            return "Montag - Freitag"

    day_list = hp.make_datetimeindex(year=year, freq="D")
    season = [_get_season(day) for day in day_list]
    day_type = [_get_day_type(day) for day in day_list]

    dt_index = hp.make_datetimeindex(year=year, freq="15min")
    ser = pd.Series(index=dt_index)

    for i, day in enumerate(day_list):
        ser.loc[day.strftime("%Y-%m-%d")] = df.loc[season[i], day_type[i]]["Wert"].values

    if freq == "60min":
        ser = hp.downsample(ser, year=str(year), aggfunc="mean")

    if peak_load is not None:
        ser = ser * (peak_load - offset) / ser.max()

    delta_T = {"15min": 0.25, "30min": 0.5, "60min": 1}[freq]

    if annual_energy is not None:
        ser = ser * (annual_energy - (offset * dt_index.size * delta_T)) / (ser.sum() * delta_T)

    ser = ser + offset

    logger.info(
        f"SLP created\n"
        f"\t{str(year)}, {freq}\n"
        f"\t{profile} ({profiles[profile]})\n"
        f"\tpeak_load: {ser.max()}\n"
        f"\tannual_energy{ser.sum() * delta_T}"
    )

    return ser


def get_TRY():
    """Experimental. Get thermal data of a Test Reference Year (TRY)."""

    fp_raw = paths.DATA / "demand/thermal/TRY_location_HSKA/TRY2045_38855002480500_Jahr.dat"

    with open(fp_raw, "r") as f:
        lines = f.readlines()

    lines_clean = lines[32:]
    del lines_clean[1]

    string = "".join(lines_clean)

    ser = pd.read_table(StringIO(string), sep=r"\s+", usecols="t", squeeze=True).drop(0)
    ser.index = ser.index - 1  # to start index with 0
    return ser


def get_ambient_temp(
    year: int, freq: str = "60min", location: str = "Rheinstetten", cache: bool = True,
) -> pd.Series:
    """OBSOLETE: Replaced by new functions to get ambient temperature.
    Experimental. Get ambient temperature from specific German stations from DWD"""

    assert year in range(2008, 2100), f"{year} is not a valid year"
    assert freq in ["60min"], f"{freq} is not a valid freq"

    fp = paths.DATA / f"weather/cache/{year}_{freq}_t_ambient_{location}.parquet"

    station_ids = {
        "Rheinstetten": "04177",
        "Kempten": "02559",
    }

    if cache and fp.exists():
        ser = read(fp)

    else:
        # German Data downloaded from
        # https://www.dwd.de/DE/leistungen/klimadatendeutschland/klarchivstunden.html?nn=16102#buehneTop
        station = station_ids[location]
        what = "hist" if year <= 2018 else "akt"
        dir_raw = paths.DATA / f"weather/DWD/stundenwerte_TU_{station}_{what}"
        fp_raw = list(dir_raw.glob("produkt_tu_stunde*.txt"))[0]
        ser = pd.read_csv(
            fp_raw, sep=";", index_col="MESS_DATUM", usecols=["MESS_DATUM", "TT_TU"], squeeze=True
        )

        ser = ser[ser.index.astype(str).str[:4] == str(year)]
        ser.reset_index(drop=True, inplace=True)
        ser.name = "t_amb"
        if cache:
            write(ser, fp)

    hp.warn_if_incorrect_index_length(ser, year, freq)
    return ser


def get_thermal_demand(
    ser_amb_temp: pd.Series,
    annual_energy: float = 1e6,
    target_temp: float = 22.0,
    threshold_temp: float = 15.0,
) -> pd.Series:
    """Returns a heating demand based on the air temperature."""
    # TODO: implement new wheather function, so that ser_amb_temp will be optional.
    assert target_temp >= threshold_temp
    ser = ser_amb_temp
    ser[ser > threshold_temp] = target_temp
    ser = target_temp - ser
    scaling_factor = annual_energy / ser.sum()
    ser *= scaling_factor
    ser.name = "Q_dem_H_T"
    logger.info(
        f"Heating demand created with annual energy={annual_energy}, target_temp={target_temp}"
        f", threshold_temp={threshold_temp}."
    )
    return ser


def get_cooling_demand(
    ser_amb_temp: pd.Series,
    annual_energy: float = 1e6,
    target_temp: float = 22.0,
    threshold_temp: float = 22.0,
) -> pd.Series:
    """Returns a cooling demand based on the ambient air temperature."""
    assert target_temp <= threshold_temp
    ser = ser_amb_temp
    ser[ser < threshold_temp] = target_temp
    ser = ser - target_temp
    scaling_factor = annual_energy / ser.sum()
    ser = ser * scaling_factor
    ser.name = "Q_dem_C_T"
    logger.info(
        f"Cooling demand created with annual energy={annual_energy}, target_temp={target_temp}"
        f", threshold_temp={threshold_temp}."
    )
    return ser


def get_PV_profile() -> pd.Series:
    """Experimental. Get a default PV profile for 1 kWh_peak."""
    # TODO: implement some PV module.
    fp = paths.DATA / "renewables/pv_el.csv"
    return read(fp=fp)
