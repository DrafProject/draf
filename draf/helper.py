import logging
import textwrap
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from elmada.helper import read, write
from numpy.core.fromnumeric import squeeze

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.CRITICAL)


def datetime_to_int(freq: str, year: int, month: int, day: int) -> int:
    """Returns the index location in a whole-year date-time-index for the given date."""
    dtindex = make_datetimeindex(year=year, freq=freq)
    i = datetime(year, month, day)
    return dtindex.get_loc(i)


def int_to_datetime(freq: str, year: int, pos: int) -> "datetime":
    dtindex = make_datetimeindex(year=year, freq=freq)
    return dtindex[pos]


def make_gif(fp: Path, duration: float = 0.5) -> None:
    """Creates a gif-file from all images in the given directory."""
    import imageio

    filenames = Path(fp).glob("*")
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename.as_posix()))
    imageio.mimsave(fp + "movie.gif", images, duration=duration)


def warn_if_incorrect_index_length(df: Union[pd.DataFrame, pd.Series], year: int, freq: str):
    len_dt = len(make_datetimeindex(year=year, freq=freq))
    len_inp = len(df)
    match = len_dt == len_inp
    if not match:
        logger.warning(f"Length mismatch: An object has {len_inp} instead of {len_dt}.")


@lru_cache(maxsize=16)
def make_datetimeindex(year: int = 2019, freq: str = "60min", tz: str = None) -> pd.DatetimeIndex:
    """Returns a whole-year date-time-index in desired resolution."""
    date_start = f"01.01.{year}  00:00:00"
    date_end = f"31.12.{year}  23:59:00"
    return pd.date_range(date_start, date_end, freq=freq, tz=tz)


def resample(
    df: Union[pd.Series, pd.DataFrame],
    year: int,
    start_freq: str,
    target_freq: Optional[str],
    aggfunc: str = "mean",
) -> Union[pd.Series, pd.DataFrame]:
    """Resamples data from a start frequency to a target frequency."""
    if (target_freq is None) or (start_freq == target_freq):
        return df
    else:
        if int_from_freq(start_freq) > int_from_freq(target_freq):
            func = upsample
        else:
            func = downsample

        return func(
            df=df, year=year, start_freq=start_freq, target_freq=target_freq, aggfunc=aggfunc
        )


def downsample(
    df: Union[pd.Series, pd.DataFrame],
    year: int,
    start_freq: str = "15min",
    target_freq: str = "60min",
    aggfunc: str = "mean",
) -> Union[pd.Series, pd.DataFrame]:
    """Downsampling for cases where start frequency is higher then target frequency.

    Args:
        df: Series or Dataframe
        year: Year
        start_freq: Time resolution of given data.
        target_freq: Time resolution of returned data.
        aggfunc:  in {"mean", "sum"}
    """

    df = df.copy()
    df.index = make_datetimeindex(year, freq=start_freq)

    resampler = df.resample(target_freq)

    if aggfunc == "sum":
        df = resampler.sum()
    elif aggfunc == "mean":
        df = resampler.mean()
    else:
        raise RuntimeError(f"aggfunc {aggfunc} not valid")

    df.reset_index(drop=True, inplace=True)

    if isinstance(df, pd.Series) and isinstance(df.name, str):
        df.name = df.name.replace(start_freq, target_freq)

    warn_if_incorrect_index_length(df, year, target_freq)
    return df


def upsample(
    df: Union[pd.Series, pd.DataFrame],
    year: int,
    start_freq: str = "60min",
    target_freq: str = "15min",
    aggfunc: str = "mean",
) -> Union[pd.Series, pd.DataFrame]:
    """Upsampling for cases where start frequency is lower then target frequency.

    Args:
        df: Series or Dataframe
        year: Year
        start_freq: Time resolution of given data.
        target_freq: Time resolution of returned data.
        aggfunc: Either 'mean' or 'sum', e.g. use `mean` for power and `sum` for aggregated energy.
    """
    df = df.copy()
    df.index = make_datetimeindex(year, freq=start_freq)
    df = df.resample(target_freq).pad()
    convert_factor = int_from_freq(start_freq) / int_from_freq(target_freq)
    if aggfunc == "sum":
        df /= convert_factor

    df.reset_index(drop=True, inplace=True)
    df = _append_rows(df, convert_factor=convert_factor)

    if isinstance(df, pd.Series) and isinstance(df.name, str):
        df.name = df.name.replace(start_freq, target_freq)

    warn_if_incorrect_index_length(df, year, target_freq)
    return df


def _append_rows(
    df: Union[pd.Series, pd.DataFrame], convert_factor: float
) -> Union[pd.Series, pd.DataFrame]:
    length = len(df)
    if isinstance(df, pd.DataFrame):
        addon_data = [df.iloc[length - 1]]
    elif isinstance(df, pd.Series):
        addon_data = df.iloc[length - 1]
    else:
        raise RuntimeError(f"unsupported type {type(df)}")

    number_of_lines = int(convert_factor - 1)
    new_index = [length + i for i in range(number_of_lines)]

    class_of_df = type(df)
    line = class_of_df(data=addon_data, index=new_index)  # new Series OR Dataframe.
    df = df.append(line).sort_index().reset_index(drop=True)
    return df


def make_quarterhourly_file_hourly(fp: Path, year: str = "2017", aggfunc: str = "mean") -> None:
    fp = Path(fp)
    ser = read(fp=fp, squeeze=True)
    sh = downsample(ser, year=year, aggfunc=aggfunc)
    filepath_new = Path(fp.as_posix().replace("15min", "60min"))
    write(sh, fp=filepath_new)


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """Returns the short version of the storage size of e.g. digital information.

    Idea from https://stackoverflow.com/questions/1094841
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:>5.1f} {unit}{suffix}"
        num /= 1000.0
    return f"{num:>5.1f} Y{suffix}"


def auto_fmt(
    num: Union[float, int, pd.Series], unit: str, target_unit: Optional[str] = None
) -> Tuple[Union[float, int, pd.Series], str]:
    """Converts data into a naturally unit if its unit starts with €, kW, or gCO2eq.

    `target_unit` is not considered if the corresponding `num` would be < 1."""

    def convert(num, unit, list, target_unit):
        for pre in list[:-1]:
            v = num.iloc[0] if isinstance(num, pd.Series) else num
            if abs(v) < 1e3 or (pre + unit) == target_unit:
                return num, pre + unit
            num /= 1e3
        return num, list[-1] + unit

    if unit.startswith("€"):
        return convert(num, unit, ["", "k", "M"], target_unit)

    elif unit.startswith("kW"):
        unit = unit[1:]
        return convert(num, unit, ["k", "M", "G", "T"], target_unit)

    elif unit.startswith("gCO2eq"):
        unit = unit[1:]
        return convert(num, unit, ["g", "kg", "t", "kt"], target_unit)

    else:
        return num, unit


def z_score(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Returns the z-score for all data points.

    Info: The z-score tells how many standard deviations below or above the population mean a raw
     score is.
    """
    return (df - df.mean()) / df.std(ddof=0)


def remove_outlier(
    df: Union[pd.Series, pd.DataFrame], zscore_threshold: float = 2.698
) -> Union[pd.Series, pd.DataFrame]:
    """Detects and removes outliers of Dataframes or Series.

    Note: The default z-score of 2.698 complies to the boxplot standard (1,5*IQR-rule).
     (see https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)
    """
    df = df.copy()

    # see
    # https://stackoverflow.com/questions/23451244/how-to-zscore-normalize-pandas-column-with-nans

    if isinstance(df, pd.DataFrame):
        for k in df:
            df[k] = remove_outlier(df[k], zscore_threshold)
        # df[(np.abs(stats.zscore(df)) > zscore_threshold).any(axis=1)] = np.nan

    if isinstance(df, pd.Series):
        outliers = np.abs(z_score(df)) > zscore_threshold
        n_outliers = outliers.sum()
        df[outliers] = np.nan
        outlier_share = n_outliers / len(df)
        outlier_treshold = 0.1
        if outlier_share > outlier_treshold:
            logger.warning(
                f"{outlier_share:.2%} (more than {outlier_treshold:.0%}) of the data were outliers"
            )
        logger.info(f"{n_outliers} datapoints where removed in {df.name}")
        # df[(np.abs(stats.zscore(df)) > zscore_threshold)] = np.nan

    return df


def optimize_plotly_layout_for_reveal_slides(
    layout: "PlotlyLayout", width: int = 750, height: int = 500
) -> "PlotlyLayout":
    """To fix this issue https://github.com/plotly/plotly.py/issues/750"""
    layout.update(autosize=False, width=width, height=height)
    return layout


def int_from_freq(freq: str) -> int:
    """E.g. '15min' -> 15"""
    return int(freq[:2])


def print_latex(df: pd.DataFrame, float_format="{:.2f}", **kwargs) -> None:
    s = df.to_latex(float_format=float_format.format, **kwargs)

    for i in [r"\toprule", r"\midrule", r"\bottomrule"]:
        s = s.replace(i, r"\hline")
    print(s)


def copy_doc(source: Callable, start: Optional[str] = None) -> Callable:
    "Copy a docstring (if present) from another source function."

    def do_copy(target):
        if source.__doc__:
            if start is None:
                target.__doc__ += source.__doc__
            else:
                loc = source.__doc__.find(start)
                target.__doc__ += "\n    " + source.__doc__[loc:]
        return target

    return do_copy


def ser_to_df_for_latex_table(ser: pd.Series, ncols: int) -> pd.DataFrame:
    nrows = len(ser) // ncols
    data = [ser[nrows * col : nrows * (col + 1)].reset_index() for col in range(ncols)]
    return pd.concat(data, 1)


def wrap_and_border(text: str, width: int) -> str:
    res = textwrap.fill(text, width=width - 2)
    return bordered(res)


def bordered(text: str) -> str:
    """Adds a border around a given text."""
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = [f"┌{'─' * width}┐"]
    for s in lines:
        res.append("│" + (s + " " * width)[:width] + "│")
    res.append(f"└{'─' * width}┘")
    return "\n".join(res)
