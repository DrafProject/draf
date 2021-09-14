import logging
import re
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


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """Returns the short version of the storage size of e.g. digital information.

    Idea from https://stackoverflow.com/q/1094841
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


def add_thousands_formatter(ax, x: bool = True, y: bool = True):
    axlist = []
    if x:
        axlist.append(ax.get_xaxis())
    if y:
        axlist.append(ax.get_yaxis())
    for axis in axlist:
        axis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))


def replace_urls_with_link(urlstr: str) -> str:
    # credits: https://stackoverflow.com/q/1112012

    pat1str = r"(^|[\n ])(([\w]+?://[\w\#$%&~.\-;:=,?@\[\]+]*)(/[\w\#$%&~/.\-;:=,?@\[\]+]*)?)"
    pat2str = r"#(^|[\n ])(((www|ftp)\.[\w\#$%&~.\-;:=,?@\[\]+]*)(/[\w\#$%&~/.\-;:=,?@\[\]+]*)?)"

    pat1 = re.compile(pat1str, re.IGNORECASE | re.DOTALL)
    pat2 = re.compile(pat2str, re.IGNORECASE | re.DOTALL)

    urlstr = pat1.sub(r'\1<a href="\2" target="_blank">\3</a>', urlstr)
    urlstr = pat2.sub(r'\1<a href="http:/\2" target="_blank">\3</a>', urlstr)

    return urlstr


def topological_sort(source):
    """perform topo sort on elements.
    credits: https://stackoverflow.com/q/11557241

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
            raise ValueError("cyclic or missing dependancy detected: %r" % (next_pending,))
        pending = next_pending
        emitted = next_emitted


def set_component_order_by_dependency(dependencies: List[Tuple[str, str]], classes: Dict) -> None:
    """Sets a order variable to classes according to its dependencies.

    Args
        deps: list of ``(name, [list of dependencies])`` pairs
        classes: A dictionary containing classes where the order argument is set to.
    """
    ordered_components_list = list(topological_sort(dependencies))
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
