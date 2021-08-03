"""A toolbox for preparation of timeseries for optimization."""
from draf.prep import param_funcs
from draf.prep.data_base import DataBase, ParDat
from draf.prep.demand import get_cooling_demand, get_el_SLP, get_thermal_demand
from draf.prep.pv import (
    get_data_for_gsee_with_data_of_nearest_station,
    get_pv_power,
    get_PV_profile,
)
from draf.prep.weather import (
    get_air_temp,
    get_ambient_temp,
    get_data_for_gsee,
    get_df_from_DWD,
    get_nearest_stations,
)
