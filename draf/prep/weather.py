import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import mpu
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from elmada.helper import read, write

from draf.paths import CACHE_DIR

DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
MIDDLE = dict(solar="/solar", air_temperature="/air_temperature/historical")

# TODO: get 10min data and resample to 15min https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/

ZIP = dict(
    solar="/stundenwerte_ST_{stations_id:05}_row.zip",
    air_temperature="/stundenwerte_TU_{stations_id:05}_{von_datum}_{bis_datum}_hist.zip",
)

DOC = dict(
    solar="/ST_Stundenwerte_Beschreibung_Stationen.txt",
    air_temperature="/TU_Stundenwerte_Beschreibung_Stationen.txt",
)

PRODUKT_TYPE = dict(solar="ST", air_temperature="TU")


def get_data_for_gsee(stations_id_air: int, stations_id_solar: int, year: int):
    """Provide solar ('global_horziontal' [W/m²], 'diffuse_fraction' [%]) and air
     ('temperature' [°C]) data for gsee, see [1].

    Note: In Germany, the mean global radiation should lie between 100 and 135 W/m², see [2].

    [1] https://gsee.readthedocs.io/en/latest/#power-output-from-a-pv-system-with-fixed-panels
    [2] https://de.wikipedia.org/w/index.php?title=Globalstrahlung&oldid=198981241#Messung_und_typische_Werte
    [3] https://www.translatorscafe.com/unit-converter/de-DE/heat-flux-density/5-1/joule/second/meter%C2%B2-watt/meter%C2%B2
    """
    at = get_df_from_DWD("air_temperature", stations_id_air)
    at.index = pd.to_datetime(at["MESS_DATUM"], format="%Y%m%d%H")
    at = at[str(year)]

    sol = get_df_from_DWD("solar", stations_id_solar)
    sol.index = pd.to_datetime(sol["MESS_DATUM_WOZ"], format="%Y%m%d%H:%M")
    sol = sol[str(year)]

    # fill values of -999 which indicate nans
    for k in ["FG_LBERG", "FD_LBERG"]:
        sol.loc[sol[k] < 0, k] = np.nan
        sol[k] = sol[k].interpolate()

    global_horizontal = sol["FG_LBERG"]
    diffuse = sol["FD_LBERG"]
    diffuse_fraction = diffuse / global_horizontal

    # convert from [J/(h·cm²)] to[W/m²]
    global_horizontal *= 10000 / 3600  # J/(h·m²)  # J/(s·m²) = W/m² , see [3] in docstring

    # fill nans due to dividing by zero
    diffuse_fraction.fillna(0, inplace=True)

    return pd.DataFrame(
        {
            "global_horizontal": global_horizontal,
            "diffuse_fraction": diffuse_fraction,
            "temperature": at["TT_TU"],
        }
    )


def get_air_temp(coords: Tuple[float, float], year: int, with_dt=False) -> pd.Series:
    """Returns air temperature for German locations."""
    data_type = "air_temperature"
    stations_id = get_nearest_station(coords=coords, data_type=data_type, year=year)["Stations_id"]
    df = get_df_from_DWD(data_type=data_type, stations_id=stations_id)
    df.index = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H")
    ser = df.loc[str(year), "TT_TU"]
    ser.index.name = "T"
    ser.name = "air_temperature"
    if not with_dt:
        ser = ser.reset_index(drop=True)
    return ser


def get_df_from_DWD(data_type: str, stations_id: int):
    """For a description of data, see [1] for solar and [2] for air temperature.

    [1] https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/solar/BESCHREIBUNG_obsgermany_climate_hourly_solar_de.pdf
    [2] https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/BESCHREIBUNG_obsgermany_climate_hourly_tu_historical_de.pdf

    data_type 'solar':
        STATIONS_ID: Identifikationsnummer der Station
        MESS_DATUM: Intervallende in UTC [yyyymmddhh:mm]
        QN_592: Qualitätsniveau der nachfolgenden Spalten [code siehe Absatz "Qualitätsinformation"]
        ATMO_LBERG: Stundensumme der atmosphärischen Gegenstrahlung [J/cm ^ 2]
        FD_LBERG: Stundensumme der diffusen solaren Strahlung [J/cm ^ 2]
        FG_LBERG: Stundensumme der Globalstrahlung [J/cm ^ 2]
        SD_LBERG: Stundensumme der Sonnenscheindauer [min]
        ZENIT: Zenitwinkel der Sonne bei Intervallmitte [Grad]
        MESS_DATUM: Intervallende in WOZ [yyyymmddhh:mm]

    data_type 'air_temperature':
        STATIONS_ID: Stationsidentifikationsnummer
        MESS_DATUM: Zeitstempel [yyyymmddhh]
        QN_9: Qualitätsniveau der nachfolgenden Spalten [code siehe Absatz "Qualitätsinformation"]
        TT_TU: Lufttemperatur in 2m Höhe [°C]
        RF_TU: relative Feuchte [%]
        eor: Ende data record
    """
    stat_id = get_stations_id_string(stations_id)
    type_id = PRODUKT_TYPE[data_type].lower()
    globlist = list(CACHE_DIR.glob(f"produkt_{type_id}_stunde_*_{stat_id}.parquet"))
    if len(globlist) == 1:
        fp = globlist[0]
        df = read(fp)
        return df
    elif len(globlist) == 0:
        unzip_and_download(data_type=data_type, stations_id=stations_id)
        return get_df_from_DWD(data_type=data_type, stations_id=stations_id)
    else:
        raise RuntimeError(f"Too many zip files for station id {stat_id}.")


def unzip_and_download(data_type: str, stations_id: int):
    url = get_zip_url(data_type=data_type, stations_id=stations_id)

    with ZipFile(BytesIO(urlopen(url).read()), "r") as myzip:
        name = get_produkt_filename_in_zip(myzip)
        fp = CACHE_DIR / f"{name}"
        fp = fp.with_suffix(".parquet")
        with myzip.open(name) as myfile:
            my_bytes = myfile.read()
            df = pd.read_csv(BytesIO(my_bytes), sep=";")
            write(df, fp)

    print(f"Cached {data_type} for station_id={stations_id} to {fp.name}")


def get_produkt_filename_in_zip(zipfile):
    produkt_files = [i.filename for i in zipfile.filelist if i.filename.startswith("produkt")]
    if len(produkt_files) == 1:
        return produkt_files[0]
    else:
        raise RuntimeError("No produkt-file in given zip-folder.")


def get_zip_file_path(data_type: str, stations_id: int):
    stat = get_stations_id_string(stations_id)
    zips = list(CACHE_DIR.glob(f"stundenwerte_{PRODUKT_TYPE[data_type]}_{stat}_*"))
    if len(zips) == 1:
        return zips[0]
    else:
        raise RuntimeError("No or too many zip files for this station id.")


def get_stations_id_string(stations_id: int) -> str:
    return f"{stations_id:05}"


def get_foldername(data_type: str, stations_id: int) -> str:
    fp = get_zip_name(data_type=data_type, stations_id=stations_id)
    return Path(fp).stem


def download_zip(data_type: str, stations_id: int):
    """DEPRECATED: currently not used in favor of 'unzip_and_download'."""
    url = get_zip_url(data_type=data_type, stations_id=stations_id)
    zipresp = urlopen(url)
    zipfilename = get_zip_name(data_type=data_type, stations_id=stations_id)
    with open(CACHE_DIR / f"{zipfilename}", "wb") as file:
        file.write(zipresp.read())


def get_zip_url(data_type: str, stations_id: int):
    zip_name = get_zip_name(data_type=data_type, stations_id=stations_id)
    return DWD_BASE + MIDDLE[data_type] + "/" + zip_name


def get_zip_name(data_type: str, stations_id: int):
    return get_zip_names(data_type=data_type)[stations_id]


def get_zip_names(data_type) -> Dict[int, str]:
    url = DWD_BASE + MIDDLE[data_type]
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    rows = soup.find_all("a")
    d = {}
    for i in rows:
        zip_name = i.get("href")
        if zip_name.startswith("stundenwerte"):
            station_id = int(zip_name.split("_")[2])
            d[station_id] = zip_name
    return d


def get_nearest_stations(coords: Tuple[float, float], year: Optional[int] = None) -> pd.DataFrame:
    types = ("solar", "air_temperature")
    d = {t: get_nearest_station(coords=coords, data_type=t, year=year) for t in types}
    return pd.DataFrame(d)


def get_nearest_station(
    coords: Tuple[float, float], data_type: str = "solar", year: Optional[int] = None
) -> pd.Series:
    assert data_type in ("solar", "air_temperature")
    df = read_stations(data_type=data_type)
    df = filter_year(df, year=year)

    lats = df["geoBreite"].values
    lons = df["geoLaenge"].values
    distance = np.zeros_like(lats)

    for i, lat, lon in zip(df.index, lats, lons):
        destination = (lat, lon)
        distance[i] = mpu.haversine_distance(origin=coords, destination=destination)

    xmin = distance.argmin()
    ser = df.loc[xmin].copy()
    ser.loc["distance_in_km"] = distance[xmin]
    return ser


def filter_year(df, year):
    if year is None:
        return df
    else:
        dt = datetime.datetime(year=year, month=12, day=31)
        bis = pd.to_datetime(df["bis_datum"], format="%Y%m%d")
        return df[bis > dt].reset_index(drop=True)


def read_stations(data_type: str, cache: bool = True) -> pd.DataFrame:
    fp_cache = CACHE_DIR / f"stations_{data_type}.parquet"

    if fp_cache.exists() and cache:
        df = read(fp_cache)

    else:
        s = read_stations_table(data_type=data_type)
        header = s.replace("\r", "").split("\n")[0]
        col_names = header.split(" ")
        df = pd.read_fwf(
            StringIO(s),
            widths=[6, 9, 8, 15, 12, 10, 42, 98],
            header=None,
            encoding="utf-8",
            skiprows=[0, 1],
            names=col_names,
        )
        write(df, fp_cache)
    return df


def read_stations_table(data_type: str) -> str:
    fp = DWD_BASE + MIDDLE[data_type] + DOC[data_type]
    return urlopen(fp).read().decode("latin-1")
