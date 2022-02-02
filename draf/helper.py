import logging
import re
import sys
import textwrap
from datetime import datetime
from functools import lru_cache
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
from geopy.geocoders import Nominatim
from matplotlib import ticker

from draf import paths

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.CRITICAL)


def fits_convention(ent_name: str, data: Union[int, float, pd.Series]) -> bool:
    """If the naming-conventions apply for the data dimensions and the entity name"""

    dims = get_dims(ent_name)
    if isinstance(data, (int, float)):
        return dims == ""
    elif isinstance(data, pd.Series):
        return data.index.nlevels == len(dims)


def get_etype(ent_name: str) -> str:
    return ent_name.split("_")[0]


def get_component(ent_name: str) -> str:
    elements = ent_name.split("_")
    return elements[1] if len(elements) >= 3 else ""


def get_desc(ent_name: str) -> str:
    elements = ent_name.split("_")
    return elements[2] if len(elements) >= 4 else ""


def get_dims(ent_name: str) -> str:
    return ent_name.split("_")[-1]


def get_step_width(freq: str) -> float:
    return int_from_freq(freq) / 60


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


def get_size(obj, seen=None):
    """Recursively finds size of objects
    from https://gist.github.com/jonathan-kosgei/90aac0b64fb345962ce3ee989fcb4075 (MIT License)
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def human_readable_size(size: float, decimal_places: int = 2):
    """based on https://stackoverflow.com/a/43690506"""

    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1e3 or unit == "PB":
            break
        size /= 1e3
    return f"{size:.{decimal_places}f} {unit}"


def conv(
    source_unit: str,
    target_unit: str,
    conversion_factor: float,
):
    """Documents unit conversion. It does nothing but to return the conversion factor."""
    return conversion_factor


def convert(
    num: Union[float, int, pd.Series], unit: str, list: List, target_unit: str = None
) -> Tuple[Union[float, int, pd.Series], str]:
    """Idea from https://stackoverflow.com/a/43690506"""
    second_part = unit[len(list[0]) :]
    for pre in list[:-1]:
        if target_unit is None:
            v = num.iloc[0] if isinstance(num, pd.Series) else num
            if abs(v) < 1e3:
                return num, pre + second_part
        else:
            if (pre + second_part) == target_unit:
                return num, pre + second_part

        num /= 1e3
    return num, list[-1] + second_part


CONV = {
    "€": [""] + list("kMGT"),
    "k€": list("kMGT"),
    "M€": list("MGT"),
    "kW": list("kMGT"),
    "kWh": list("kMGT"),
    "MW": list("MGT"),
    "MWh": list("MGT"),
    "gCO2eq": ["g", "kg", "t", "kt"],
    "kgCO2eq": ["kg", "t", "kt", "Mt", "Gt"],
    "tCO2eq": ["t", "kt", "Mt", "Gt"],
}


def auto_fmt(
    num: Union[float, int, pd.Series], unit: str, target_unit: Optional[str] = None
) -> Tuple[Union[float, int, pd.Series], str]:
    """Converts data into a naturally unit if its unit starts with €, kW, or gCO2eq.
    WARNING: Only specific units available.
    `target_unit` is not considered if the corresponding `num` would be < 1.
    """
    first = unit.split("/")[0].split("_")[0]
    if first in CONV:
        return convert(num, unit, CONV[first], target_unit)
    else:
        return num, unit


def auto_convert_units_colwise(df, units: Dict):
    for name, unit in units.items():
        df[name], units[name] = auto_fmt(
            num=df[name], unit=unit, target_unit=auto_fmt(df[name].sum(), unit)[1]
        )
    return df, units


def consider_timestepdelta(df, units: Dict, time_step_width: float):
    for name, unit in units.items():
        if "T" in get_dims(name):
            df[name] = df[name] * time_step_width
            units[name] = aggregate_unit(unit)
    return df, units


def aggregate_unit(unit: str) -> str:
    parts = unit.split("_")
    first = parts[0]
    if first in ["kW", "MW", "GW", "TW"]:
        first = first.replace("W", "Wh")
    return "_".join([first] + parts[1:])


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


def add_thousands_formatter(ax, x: bool = True, y: bool = True):
    axlist = []
    if x:
        axlist.append(ax.get_xaxis())
    if y:
        axlist.append(ax.get_yaxis())
    for axis in axlist:
        axis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))


def replace_urls_with_link(urlstr: str) -> str:
    """credits: https://stackoverflow.com/a/1112670"""

    pat1str = r"(^|[\n ])(([\w]+?://[\w\#$%&~.\-;:=,?@\[\]+]*)(/[\w\#$%&~/.\-;:=,?@\[\]+]*)?)"
    pat2str = r"#(^|[\n ])(((www|ftp)\.[\w\#$%&~.\-;:=,?@\[\]+]*)(/[\w\#$%&~/.\-;:=,?@\[\]+]*)?)"

    pat1 = re.compile(pat1str, re.IGNORECASE | re.DOTALL)
    pat2 = re.compile(pat2str, re.IGNORECASE | re.DOTALL)

    urlstr = pat1.sub(r'\1<a href="\2" target="_blank">\3</a>', urlstr)
    urlstr = pat2.sub(r'\1<a href="http:/\2" target="_blank">\3</a>', urlstr)

    return urlstr


def topological_sort(source):
    """perform topo sort on elements.
    credits: https://stackoverflow.com/a/11564769

    :arg source: list of ``(name, [list of dependancies])`` pairs
    :returns: list of names, with dependancies listed first
    """
    pending = [
        (name, set(deps)) for name, deps in source
    ]  # copy deps so we can modify set in-place
    emitted = []
    while pending:
        next_pending = []
        next_emitted = []
        for entry in pending:
            name, deps = entry
            deps.difference_update(emitted)  # remove deps we emitted last pass
            if deps:  # still has deps? recheck during next pass
                next_pending.append(entry)
            else:  # no more deps? time to emit
                yield name
                emitted.append(name)  # <-- not required, but helps preserve original ordering
                next_emitted.append(
                    name
                )  # remember what we emitted for difference_update() in next pass
        if not next_emitted:  # all entries have unmet deps, one of two things is wrong...
            raise ValueError("cyclic or missing dependency detected: %r" % (next_pending,))
        pending = next_pending
        emitted = next_emitted


def set_component_order_by_order_restrictions(
    order_restrictions: List[Tuple[str, str]], classes: Dict
) -> None:
    """Sets an order variable to classes according to its order_restrictions.

    Args
        order_restrictions: list of ``(name, [list of order_restrictions])`` pairs
        classes: A dictionary containing classes where the order argument is set to.
    """
    ordered_components_list = list(topological_sort(order_restrictions))
    for i, classname in enumerate(ordered_components_list):
        classes[classname].order = i


def make_symlink_to_cache():
    """Creates a symbolic link to the cache directory for easy access.

    Note: This function requires admin privileges.
    """
    link_dir = paths.BASE_DIR / "cache"
    cache_dir = paths.CACHE_DIR
    link_dir.symlink_to(target=cache_dir, target_is_directory=True)
    print(f"Symbolic link created: {link_dir} --> {cache_dir}")


def get_value_from_varOrPar(term):
    try:
        return term.x  # gurobipy.Var
    except AttributeError:
        try:
            return term.getValue()  # gurobipy.LinExpr
        except AttributeError:
            try:
                return term.value  # pyomo.environ.Var
            except AttributeError:
                assert isinstance(term, (float, int)), f"type: {type(term)}"
                return term  # float, int


def get_annuity_factor(r: float = 0.06, N: float = 20):
    """Returns the annuity factor of a given return rate r and an expected lifetime N"""
    return (r * (1 + r) ** N) / ((1 + r) ** N - 1)


@lru_cache(maxsize=50)
def _get_address(address: str, user_agent: str) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent=user_agent)
    location = geolocator.geocode(address)
    return location


def address2coords(address: str, user_agent: str = "anonymous_draf_user") -> Tuple[float, float]:
    """Returns geo coordinates (latitude, longitude) from given address."""
    location = _get_address(address=address, user_agent=user_agent)
    if location is None:
        raise RuntimeError("No location found. Please try a different address.")
    else:
        coords = (location.latitude, location.longitude)
        print(f"Used location: {location.address}, {coords}")
        return coords


def delete_cache(filter_str: str = "*") -> None:
    """Deletes parts or all of the cache directory.

    Unless `filter_str` is '*' only files are selected that contain the `filter_string` somewhere
    in the filename.
    """

    s = "*" if filter_str == "*" else f"*{filter_str}*"

    files = list(paths.CACHE_DIR.glob(f"{s}"))
    lenf = len(files)

    if lenf == 0:
        print(f"No file found containing '{filter_str}'.")

    else:
        print(f"{lenf} files containing '{filter_str}':")

        for f in files:
            size = human_readable_size(f.stat().st_size)
            print(f"\t{size:>5}  {f.name}")

        if confirm_deletion(lenf):
            for f in files:
                f.unlink()
            print(f"{lenf} files deleted")

        else:
            print("No files deleted")


def confirm_deletion(nfiles: int) -> bool:
    return input(f"Do you really want to delete these {nfiles} files? (y/n)") == "y"
