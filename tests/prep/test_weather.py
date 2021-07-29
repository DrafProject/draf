import pandas as pd
import pytest

from draf.prep import weather


def test_get_data_for_gsee():
    df = weather.get_data_for_gsee(stations_id_air=4063, stations_id_solar=691, year=2019)
    mean = df["global_horizontal"].mean()
    assert 90 < mean < 150, f"Average global radiation data (={mean:.3f}) are unrealistic."


def test_get_nearest_stations():
    df = weather.get_nearest_stations(year=2019, coords=(49.01, 8.39))
    assert isinstance(df, pd.DataFrame)
    assert tuple(df.iloc[6].values) == ("Mannheim", "Rheinstetten")


def test_read_stations():
    assert isinstance(weather.read_stations("solar"), pd.DataFrame)
    assert isinstance(weather.read_stations("air_temperature"), pd.DataFrame)


@pytest.mark.slow
def test_read_stations_table():
    assert isinstance(weather.read_stations_table("solar"), str)
    assert isinstance(weather.read_stations_table("air_temperature"), str)


def test_get_air_temp():
    assert weather.get_air_temp(year=2019, coords=(49.01, 8.39)).mean() == pytest.approx(11.766061)
