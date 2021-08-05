import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from elmada.helper import (
    estimate_freq,
    int_from_freq,
    make_datetimeindex,
    read,
    remove_outlier,
    resample,
    warn_if_incorrect_index_length,
    write,
    z_score,
)

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


def make_quarterhourly_file_hourly(fp: Path, year: str = "2017", aggfunc: str = "mean") -> None:
    fp = Path(fp)
    ser = read(fp=fp, squeeze=True)
    sh = resample(ser, year=year, start_freq="15min", target_freq="60min", aggfunc=aggfunc)
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


def optimize_plotly_layout_for_reveal_slides(
    layout: "PlotlyLayout", width: int = 750, height: int = 500
) -> "PlotlyLayout":
    """To fix this issue https://github.com/plotly/plotly.py/issues/750"""
    layout.update(autosize=False, width=width, height=height)
    return layout


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


def check(name: str, check: bool):
    check_res = "✅" if check else "⛔"
    print(f"{check_res} {name}")
