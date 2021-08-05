import logging
import textwrap
from io import StringIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as po
import seaborn as sns
from elmada import plots
from IPython.display import display
from matplotlib.colors import Colormap, DivergingNorm, LinearSegmentedColormap
from plotly.tools import make_subplots

from draf import helper as hp
from draf.plotting.base_plotter import BasePlotter

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)

COLORS = {
    "E": "hsla(120, 50%, 70%, 0.7)",
    "F": "hsla(240, 50%, 70%, 0.7)",
    "Q": "hsla(10, 50%, 70%, 0.7)",
}


class ScenPlotter(BasePlotter):
    """Plotter for scenarios.

    Args:
        sc: The scenario object containing all data.
    """

    def __init__(self, sc):
        self.figsize = (16, 4)
        self.sc = sc
        self.cs = sc._cs
        self.notebook_mode: bool = self.script_type() == "jupyter"
        self.optimize_layout_for_reveal_slides = True

    def __getstate__(self):
        """Remove objects with dependencies for serialization with pickle."""
        d = self.__dict__.copy()
        d.pop("sc", None)
        return d

    def display(self, what: str = "p"):
        """Display all entities unstacked to the first dimensions.

        Args:
            what: 'v' for Variables, 'p' for Parameters.
        """
        sc = self.sc
        dims_dic = sc._get_entity_store(what=what)._to_dims_dic(unstack_to_first_dim=1)
        for dim, data in dims_dic.items():
            if dim == "":
                data = pd.Series(data).to_frame(name="Scalar value")
            max_rows = 300 if dim == "" else 24
            with pd.option_context("display.max_rows", max_rows):
                display(data)

    def heatmap_py(
        self,
        timeseries: Union[np.ndarray, pd.Series] = None,
        ent_name: str = None,
        title: Optional[str] = None,
        show_title: bool = True,
        cmap: str = "OrRd",
        colorbar_label: str = "",
    ) -> go.Figure:
        """Returns a Plotly heatmap of a given timeseries or entity name."""
        if isinstance(timeseries, np.ndarray):
            timeseries = pd.Series(timeseries)

        if ent_name is not None and timeseries is None:
            timeseries = self.sc.get_entity(ent=ent_name)
            colorbar_label = f"{colorbar_label} [{self.sc.get_unit(ent_name)}]"
            title = f"{ent_name}: {self.sc.get_doc(ent_name)}"

        layout = go.Layout(
            title=title,
            xaxis=dict(title=f"Days of {self.cs.year}"),
            yaxis=dict(title="Hours of day"),
        )

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        data = timeseries.values.reshape((self.cs.steps_per_day, -1), order="F")[:, :]
        idx = self.cs.dated(timeseries).index
        data = go.Heatmap(
            x=pd.date_range(start=idx[0], end=idx[-1], freq="D"),
            z=data,
            colorbar=dict(title=colorbar_label),
            colorscale=cmap,
        )

        fig = go.FigureWidget(layout=layout, data=data)
        return fig

    def heatmap_line_py(
        self,
        timeseries: Union[np.ndarray, pd.Series] = None,
        ent_name: str = None,
        title: Optional[str] = None,
        cmap: str = "OrRd",
        colorbar_label: Optional[str] = None,
    ) -> go.Figure:
        """Returns a combined heatmap-line Plotly plot of a given timeseries or entity name."""

        if isinstance(timeseries, np.ndarray):
            ser = pd.Series(timeseries)

        elif ent_name is not None:
            ser = self.sc.get_entity(ent=ent_name)
            colorbar_label = f"{ent_name} [{self.sc.get_unit(ent_name)}]"
            title = f"{ent_name}: {self.sc.get_doc(ent_name)} [{self.sc.get_unit(ent_name)}]"

        else:
            raise Exception("No timeseries specified!")

        data = ser.values.reshape((self.cs.steps_per_day, -1), order="F")[:, :]
        idx = self.cs.dated(ser).index

        trace1 = go.Scatter(x=idx, y=ser.values, line_width=1)

        trace2 = go.Heatmap(
            x=pd.date_range(start=idx[0], end=idx[-1], freq="D"),
            z=data,
            colorbar=dict(title=colorbar_label, titleside="right"),
            colorscale=cmap,
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            specs=[[{}], [{}]],
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.13,
            subplot_titles=(None, None),
        )

        fig.update_yaxes(title_text=colorbar_label, row=1, col=1)
        fig.update_xaxes(title_text=f"Time [{self.cs._freq_unit}]", row=1, col=1)
        fig.update_yaxes(title_text=f"Time [{self.cs._freq_unit}]", row=2, col=1)
        fig.update_xaxes(title_text=f"Days of {self.cs.year}", row=2, col=1)

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 2, 1)
        fig.update_layout(title_text=title)
        return fig

    def heatmap(
        self,
        timeseries: Union[np.ndarray, pd.Series] = None,
        ent_name: str = None,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        colorbar_label: str = "",
        show_cbar: bool = True,
        cmap: str = "OrRd",
        show_title: bool = True,
        text: Optional[str] = None,
        text_alpha: float = 0.2,
        yaxis_factor: float = 2.0,
        divergingNorm: bool = True,
        **imshow_kws,
    ) -> Tuple["fig", "ax"]:
        """Plots a matplotlib heatmap of a given timeseries or entity name..

        Args:
            timeseries: Data to be plotted.
            ent_name: Entity name to be plotted.
            title: Custom title.
            suptitle: Bigger title.
            colorbar_label: The label of the colorbar.
            show_cbar: If the color bar is shown.
            cmap: Specify the color map.
            show_title: If the title is shown.
            text: If a provided, a string overlay is placed on the heatmap.
            text_alpha: Alpha value of the string overlay.
            yaxis_factor: A float factor, that scales that y-value of the existing figsize tuple.
            divergingNorm: If the diverging colorrange should be normed so that zero is white.
            **imshow_kws: Keyword arguments for matplotlib imshow.

        """

        if isinstance(timeseries, np.ndarray):
            timeseries = pd.Series(timeseries)

        if timeseries is not None:
            series = timeseries

        elif ent_name is not None:
            series = self.sc.get_entity(ent=ent_name)
            colorbar_label = f"{colorbar_label} [{self.sc.get_unit(ent_name)}]"
            if title is None:
                title = f"{ent_name}: {self.sc.get_doc(ent_name)}"

        else:
            raise Exception("No timeseries specified.")

        fig, ax, img = self._get_core_heatmap_fig(
            series=series,
            cmap=cmap,
            yaxis_factor=yaxis_factor,
            divergingNorm=divergingNorm,
            **imshow_kws,
        )
        if show_title:
            ax.set_title(title)
            fig.suptitle(suptitle, fontsize=14, weight="bold")

        if show_cbar:
            cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
            fig.colorbar(img, cax=cbar_ax, label=colorbar_label)

        if text is not None:
            font = {
                "family": "sans-serif",
                "color": "black",
                "weight": "bold",
                "size": 60,
                "alpha": text_alpha,
            }
            plt.text(
                0.5,
                0.5,
                text,
                fontdict=font,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        return fig, ax

    def sankey(self, string_builder_func, title="") -> go.Figure:
        """Returns a Plotly sankey plot.

        Args:
            string_builder_func: Function that returns a string in the form of:
                type source target value
                F GAS CHP 1000
                E CHP EL 450
        """
        fig = self.get_sankey_fig(string_builder_func=string_builder_func, title=title)

        self._plot_plotly_fig(fig, filename=self.cs._res_fp / f"{title}.html")
        return fig

    def line(
        self, ent_name: Optional[str] = None, data: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> go.Figure:
        """Returns a Plotly line plot of one entitiy name or one series.

        Args:
            ent_name: Entity name to be plotted.
            data: Data to be plotted.
        """

        assert ent_name is not None or data is not None, "Please provide ent_name or data."

        if ent_name is not None:
            data = self.sc.get_entity(ent_name)

            if self.sc._get_dims(ent_name) == "T":
                data = self.cs.dated(data)

        assert isinstance(data, pd.Series)
        return self._get_line_fig(data)

    def _get_line_fig(self, data: Union[pd.Series, pd.DataFrame]) -> go.Figure:

        # TODO: check dimensions and unstack if needed to cope with multidimensional data.

        df = _convert_to_df(data)

        plotly_data = []
        for _, ser in df.items():
            trace = go.Scatter(x=ser.index, y=ser.values)
            plotly_data.append(trace)

        return go.Figure(data=plotly_data)

    def line_T(self, what: str = "v", dated: bool = True) -> go.Figure:
        """Returns a Plotly line plot of all entities with the dimension T using Plotly express.

        Attention: the figure might be very big.

        Args:
            what: 'v' for Variables, 'p' for Parameters.
            dated: If index has datetimes.
        """
        sc = self.sc
        cs = sc._cs

        df = sc.get_var_par_dic(what)["T"]

        if dated:
            df = cs.dated(df)

        ser = df.stack()
        ser.index = ser.index.rename(["T", "ent"])
        ser = ser.rename("values")
        df = ser.reset_index()
        df["flow_type"] = df.ent.apply(lambda x: x.split("_")[0]).astype("category")
        df["ent"] = df["ent"].astype("category")

        y_scaler = len(df.flow_type.unique())
        fig = px.line(
            df, x="T", y="values", color="ent", facet_row="flow_type", height=200 * y_scaler
        )
        return fig

    def plot(
        self,
        which_ents: Optional[Dict[str, List[str]]] = None,
        which_dims: Optional[str] = None,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        use_dtindex: bool = True,
        what: str = "v",
        **pltargs,
    ) -> None:
        """EXPERIMENTAL and buggy.

        Returns Plotly line plots of all desired entities with Matplotlib.

        Args:
            which_ents: A dictionary which dimensions as keys and as values, desired entities to plot.
            which_dims: Specify which dimensions to plot. e.g. 'TS'
            t_start: Start time step
            t_end: End time step
            what: 'v' for Variables, 'p' for Parameters.
            **pltargs: e.g.: drawstyle = 'steps-post' or kind = 'bar'

        Example:
            ```
            sc.plot.plot(which_ents={'T': ['E_BES_T'], 'TS': ['G_STO_TS']}, linestyle='steps')
            ```
        """
        sc = self.sc
        # TODO: refactor: split in smaller functions
        dims_dic = self.sc.get_var_par_dic(what)

        pltargs["figsize"] = self.figsize

        if which_dims is not None:
            assert isinstance(which_dims, str)
            which_ents = {which_dims: list(dims_dic[which_dims].columns)}

        if which_ents is None:
            which_ents = self._get_all_plottable_entities_from(dims_dic)

        if which_ents is not None:
            if isinstance(which_ents, str):
                which_ents = [which_ents]

        assert isinstance(which_ents, dict)

        # TODO: plot all with the same unit when argument consider_units is switched on

        for dims, ent_name in which_ents.items():

            dim_str = ", ".join(list(dims))
            title_dim = f"{self.sc.name}: dims {dim_str}"

            y_dtindex = use_dtindex and (dims[0] == "T")

            if len(dims) == 1:
                data = sc.get_entity(ent_name)

                if t_end is None:
                    data = self.cs.dated(dims_dic[dims][which_ents[dims]], y_dtindex)
                    data.plot(title=title_dim, **pltargs)
                else:
                    data = self.cs.dated(
                        dims_dic[dims][which_ents[dims]][:][t_start:t_end], y_dtindex
                    )
                    data.plot(title=title_dim, **pltargs)

            elif len(dims) == 2:
                for ent in which_ents[dims]:
                    title_ent = f"{self.sc.id}: {ent} ({dim_str})"
                    if t_end is None:
                        data = self.cs.dated(dims_dic[dims][ent].unstack(), y_dtindex)
                        data.plot(title=title_ent, **pltargs)
                    else:
                        data = self.cs.dated(
                            dims_dic[dims][ent][t_start:t_end].unstack(), y_dtindex
                        )
                        data.plot(title=title_ent, **pltargs)

            elif len(dims) >= 3:
                for ent in which_ents[dims]:
                    title_ent = f"{self.sc}: {ent} ({dim_str})"
                    if t_end is None:
                        data = self.cs.dated(dims_dic[dims][ent].unstack().unstack(), y_dtindex)
                        data.plot(title=title_ent, **pltargs)
                    else:
                        data = self.cs.dated(
                            dims_dic[dims][ent][t_start:t_end].unstack().unstack(), y_dtindex
                        )
                        data.plot(title=title_ent, **pltargs)

    def _get_all_plottable_entities_from(self, dims_dic) -> Dict[str, str]:
        return {dims: list(dims_dic[dims].columns) for dims in dims_dic if dims != ""}

    def _get_sankey_df(self, string_builder_func: Callable) -> pd.DataFrame:
        assert callable(string_builder_func)
        string = textwrap.dedent(string_builder_func(self.sc))
        df = pd.read_csv(StringIO(string), sep=" ")
        df["value"] /= 1e3
        return df

    def get_sankey_fig(self, string_builder_func: Callable, title: str = "") -> go.Figure:
        """Returns Plotly Sankey figure.

        Args:
            string_builder_func: Function that returns a string in the form of:
                type source target value
                F GAS CHP 1000
                E CHP EL 450
            title: Title of the Sankey figure.
        """

        df = self._get_sankey_df(string_builder_func)
        source_s, target_s, value = (list(df[s]) for s in ["source", "target", "value"])

        label = list(set(source_s + target_s))
        source = [label.index(x) for x in source_s]
        target = [label.index(x) for x in target_s]

        link_color = [COLORS[x] for x in df["type"].values.tolist()] if COLORS is not None else None

        data = dict(
            type="sankey",
            arrangement="snap",
            orientation="h",
            valueformat=".2f",
            valuesuffix="MWh",
            node=dict(
                pad=10,
                thickness=10,
                line=dict(color="white", width=0),
                label=label,
                color="hsla(0, 0%, 0%, 0.5)",
            ),
            link=dict(source=source, target=target, value=value, color=link_color),
        )

        layout = dict(title=title, font=dict(size=15), margin=dict(t=35, b=5, l=5, r=5))

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        return go.Figure(data=[data], layout=layout)

    def merit_order(self, **kwargs) -> Tuple["fig", "ax", "ax_right"]:
        return plots.merit_order(year=self.cs.year, country=self.cs.country, **kwargs)

    def describe(
        self,
        filter_str: str = "",
        sort_values: bool = True,
        include_pars: bool = True,
        include_vars: bool = True,
        natural_units: bool = False,
        make_plots: bool = False,
        log_scale: bool = True,
    ) -> None:
        """Prints a description of all parameters and results of the scenario with its units.
        Scalars are shown by its value. Multi-dimensional data are described by sum, mean, min,
        max.

        Args:
            filter_str (str): Only entities containing this string are considered. Good to filter
                components e.g. 'CHP' or entity types e.g. 'c_'.
            sort_values (bool): If values are sorted. Otherwise, entities are sorted alphanumerical.
            include_pars (bool): If parameters are considered.
            include_vars (bool): If variable results are considered.
            natural_units (bool): If units are converted to natural units.
            make_plots (bool): If plots are shown.
            log_scale (bool): If plots have logarithmic scale.
        """
        entTypeDic = self._get_filled_entTypeDic(include_pars, include_vars)

        for entType, dims_dic in entTypeDic.items():
            desc_dict = self._make_descDict_and_plot_data(
                dims_dic, filter_str, make_plots, sort_values, log_scale
            )

            self._pretty_print_descDict(entType, desc_dict, natural_units)
            # TODO: display an interactive html-version with <details> tags.

    def _make_descDict_and_plot_data(
        self, dims_dic, filter_str, make_plots, sort_values, log_scale
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        desc_dict = {}
        for dim, data in dims_dic.items():
            if _dim_contains_valid_ents(data=data, filter_str=filter_str):
                desc_dict[dim] = {}
            else:
                continue

            if dim == "":
                assert isinstance(data, Dict)
                filtered_scalars = {k: v for k, v in data.items() if filter_str in k}
                ser = pd.Series(filtered_scalars)
                ser = ser.sort_values(ascending=False) if sort_values else ser.sort_index()

                desc_dict[dim] = ser

                if make_plots:
                    self._plot_scalar_descriptions(scalar_ser=ser, log_scale=log_scale)
            else:
                assert isinstance(data, pd.DataFrame)
                df_desc = self._get_df_desc(
                    original_df=data, sort_values=sort_values, filter_str=filter_str
                )
                desc_dict[dim] = df_desc.copy()

                if make_plots:
                    self._plot_nonscalar_descriptions(df_desc, log_scale, dim)

        return desc_dict

    def _plot_scalar_descriptions(self, scalar_ser: pd.Series, log_scale: bool) -> None:
        y_scaling = len(scalar_ser) / 16
        figsize = (self.figsize[0], self.figsize[1] * y_scaling)
        _, ax = plt.subplots(figsize=figsize)
        scalar_ser.iloc[::-1].plot.barh(color="cornflowerblue", log=log_scale, ax=ax)
        ax.set(xlabel="Scalar values")
        sns.despine()

    def _plot_nonscalar_descriptions(
        self, df_desc: pd.DataFrame, log_scale: bool, dim: str
    ) -> None:
        y_scaling = len(df_desc.index) / 20
        figsize = (self.figsize[0], self.figsize[1] * y_scaling)
        _, ax = plt.subplots(figsize=figsize)
        df_desc["sum"].iloc[::-1].plot.barh(color="grey", logx=log_scale, ax=ax)
        ax.set(xlabel=f"Sum of dim={dim}")
        sns.despine()

    def _get_df_desc(
        self, orig_df: pd.DataFrame, sort_values: bool, filter_str: str
    ) -> pd.DataFrame:
        df = orig_df[[col for col in orig_df if filter_str in col]]
        df_desc = df.describe(percentiles=[], include=np.number).transpose()

        for k in ("50%", "count", "std"):
            df_desc.pop(k)

        df_desc.insert(loc=0, column="sum", value=[df[x].values.sum() for x in list(df_desc.index)])

        if sort_values:
            df_desc = df_desc.sort_values(by="sum", ascending=False)
        else:
            df_desc = df_desc.sort_index()

        df_desc[""] = [str(self.sc.get_unit(name)).replace("None", "") for name in df_desc.index]
        return df_desc

    def _get_filled_entTypeDic(self, include_pars, include_vars) -> Dict[str, Dict]:
        d = {}

        if include_pars and self.sc.par_dic is not None:
            d["Parameters"] = self.sc.par_dic

        if include_vars and self.sc.res_dic is not None:
            d["Variables"] = self.sc.res_dic

        return d

    def _pretty_print_descDict(self, what_type: str, desc_dict: Dict, natural_units: bool) -> None:
        header = f"{what_type.capitalize()} for {self.sc.id} (doc='{self.sc.doc}')"
        print(hp.bordered(header))
        for dim, val in desc_dict.items():
            print(hp.bordered(f"_{dim}" if dim != "" else "_ (Scalars)"))

            if dim == "":
                for name, num in val.items():
                    unit = self.sc.get_unit(name)

                    if natural_units:
                        num, unit = hp.auto_fmt(num, unit)
                    print("{:<15}{:>15,.4f} {}".format(name, num, unit))

            else:
                print(val)
            print()

    def _plot_plotly_fig(self, fig, **kwargs):
        if self.notebook_mode:
            po.init_notebook_mode(connected=True)
            po.iplot(fig, **kwargs)
        else:
            po.plot(fig, **kwargs)

    def _get_core_heatmap_fig(
        self, series, cmap, yaxis_factor: float, divergingNorm: bool = True, **imshow_kws
    ) -> Tuple:
        steps_per_day = self.cs.steps_per_day

        assert len(series) % steps_per_day == 0, (
            f"Timeseries doesn't fit the steps per day. There are "
            f"{steps_per_day - len(series) % steps_per_day:.0f} timesteps missing."
            f" (timeseries: {len(series):.0f}, steps_per_day:{steps_per_day:.0f})"
        )

        data = series.values.reshape((steps_per_day, -1), order="F")

        hours = np.arange(steps_per_day, -1, -4)
        x_lims = mdates.date2num(self.sc.dtindex_custom.to_pydatetime())
        y_lims = hours

        figsize = (self.figsize[0], self.figsize[1] * yaxis_factor)

        if "ax" in imshow_kws:
            fig = plt.gcf()
            ax = imshow_kws.pop("ax")
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.subplots_adjust(right=0.85, hspace=0.1)

        if divergingNorm:
            cmap = _make_diverging_norm(series, imshow_kws)

        imshow_kws.update(cmap=cmap)

        img = ax.imshow(
            data,
            extent=[x_lims[0], x_lims[-1], y_lims[0], y_lims[-1]],
            interpolation="none",
            aspect="auto",
            **imshow_kws,
        )

        ax.set(yticks=hours)
        ax.set_ylabel(f"Time [{self.cs._freq_unit}]")
        ax.set_xlabel("Time [Days]", labelpad=10)
        ax.xaxis_date()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        date_format = mdates.DateFormatter("%b")
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(direction="out")

        return fig, ax, img


def _make_diverging_norm(ser, imshow_kws) -> Colormap:
    colors = ["white", "indianred", "darkred"]

    if ser.min() < 0 < ser.max():
        imshow_kws.update(norm=DivergingNorm(vmin=ser.min(), vcenter=0.0, vmax=ser.max()))
        colors = ["darkblue", "steelblue"] + colors

    return LinearSegmentedColormap.from_list("_cmap", colors=colors)


def _dim_contains_valid_ents(data: Union[Dict, pd.DataFrame], filter_str: str) -> bool:
    if isinstance(data, dict):
        return bool([k for k in data if filter_str in k])
    elif isinstance(data, pd.DataFrame):
        return bool([k for k in data if filter_str in k])


def _convert_to_df(data=Union[np.ndarray, pd.Series, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    assert isinstance(data, pd.DataFrame)
    return data
