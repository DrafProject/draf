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
from matplotlib.colors import Colormap, LinearSegmentedColormap, TwoSlopeNorm
from plotly.subplots import make_subplots

from draf import helper as hp
from draf.conventions import Etypes
from draf.plotting.base_plotter import BasePlotter

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)

COLORS = {
    "E": "hsla(120, 50%, 70%, 0.7)",  # electricity
    "F": "hsla(240, 50%, 70%, 0.7)",  # fuel
    "Q": "hsla(10, 50%, 70%, 0.7)",  #  high temperature thermal energy
    "C": "hsla(209, 57%, 84%, 0.7)",  # low temperature thermal energy
    "M": "hsla(24, 100%, 80%, 0.7)",  # medium temperature thermal energy
    "H": "hsla(250, 83%, 56%, 0.7)",  # hydrogen
}


class ScenPlotter(BasePlotter):
    """Plotter for scenarios.

    Args:
        sc: The scenario object containing all data.
    """

    def __init__(self, sc):
        self.figsize = (16, 4)
        self.sc = sc
        self.notebook_mode: bool = self.script_type() == "jupyter"
        self.optimize_layout_for_reveal_slides = False

    def __getstate__(self):
        """Remove objects with dependencies for serialization with pickle."""
        d = self.__dict__.copy()
        d.pop("sc", None)
        return d

    def display(self, what: str = "p"):
        """Display all entities unstacked to the first dimensions.

        Args:
            what: 'v' for variables, 'p' for parameters.
        """
        dims_dic = self.sc._get_entity_store(what=what)._to_dims_dic(unstack_to_first_dim=True)
        for dim, data in dims_dic.items():
            if dim == "":
                data = pd.Series(data).to_frame(name="Scalar value")
            max_rows = 300 if dim == "" else 24
            with pd.option_context("display.max_rows", max_rows):
                display(data)

    def collector_table(
        self, font_size: float = 7.0, divide_all_numbers_by: float = 1.0, gradient: bool = False
    ):
        """Return table with collectors and components."""
        sc = self.sc
        df = pd.DataFrame(sc.collector_values).div(divide_all_numbers_by).T.sort_index(axis=1)
        df = df[df.count().sort_values(ascending=False).index]
        df.index = pd.MultiIndex.from_tuples(
            [(i, sc.get_unit(i)) for i in df.index], names=["Name", "Unit"]
        )
        s = (
            df.style.format("{:,.0f}", na_rep="")
            .applymap(lambda x: "background-color: transparent" if pd.isnull(x) else "")
            .set_table_styles(
                [
                    {"selector": "th", "props": f"font-size:{font_size}pt;"},
                    {"selector": "td", "props": f"font-size:{font_size}pt;"},
                ]
            )
        )
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        return s

    def correlation_table(self, gradient: bool = False, caption: bool = False):
        df = self.sc.get_flat_T_df().corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
        df.columns = [str(i) for i in range(len(df.columns))]
        df.index = [f"{c}_{i}" for c, i in zip(df.columns, df.index)]
        s = df.style.format("{:.2f}", na_rep="").set_sticky(axis=0).set_sticky(axis=1)
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        if caption:
            s = s.set_caption("Table of Pearson correlation factors for all time series.")
        return s

    def collectors(
        self,
        filter_etype: Optional[str] = None,
        auto_convert_units: bool = True,
        consider_stepwidth: bool = True,
        use_plt: bool = False,
    ):
        sc = self.sc
        df = pd.DataFrame(sc.get_all_collector_values())

        units = {name: sc.get_unit(name) for name in df}

        if consider_stepwidth:
            df, units = hp.consider_timestepdelta(df, units, time_step_width=sc.step_width)

        if auto_convert_units:
            df, units = hp.auto_convert_units_colwise(df, units)

        df = (
            df.T.stack()
            .reset_index()
            .rename(columns={"level_0": "collector", "level_1": "comp", 0: "value"})
        )
        if filter_etype is not None and filter_etype != "":
            df = df[df["collector"].apply(lambda s: s.split("_")[0] == filter_etype)]

        df["doc"] = df["collector"].apply(sc.get_doc)
        # df["etype"] = df["collector"].apply(hp.get_etype)
        df["unit"] = df["collector"].replace(units)
        df["desc"] = "<b>" + df["collector"] + "</b> - " + df["doc"] + " (" + df["unit"] + ")"

        if use_plt:
            self._plt_collector_plot(df)
        else:
            return self._plotly_collector_plot(df)

    def _plt_collector_plot(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.tight_layout()
        df = df.pivot(index="desc", columns="comp", values="value")
        df.plot.barh(stacked=True, ax=ax, width=0.8)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set_ylabel("")
        ax.invert_yaxis()
        sns.despine()

    def _plotly_collector_plot(self, df):
        return (
            px.bar(
                df,
                y="desc",
                x="value",
                color="comp",
                # pattern_shape="comp",  #FIXME: may cause issues
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                category_orders={"desc": sorted(df.desc.unique())},
            )
            .update_yaxes(title="")
            .update_xaxes(title="")
        ).update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0})

    def heatmap_py(
        self,
        timeseries: Union[np.ndarray, pd.Series] = None,
        ent_name: str = None,
        title: Optional[str] = None,
        cmap: str = "OrRd",
        colorbar_label: str = "",
    ) -> go.Figure:
        """Returns a Plotly heatmap of a given timeseries or entity name."""
        if isinstance(timeseries, np.ndarray):
            timeseries = pd.Series(timeseries)

        if ent_name is not None and timeseries is None:
            timeseries = self.sc.get_entity(ent=ent_name)
            colorbar_label = f"{colorbar_label} ({self.sc.get_unit(ent_name)})"
            if title is None:
                title = f"{ent_name}: {self.sc.get_doc(ent_name)}"

        layout = go.Layout(
            title=title,
            xaxis=dict(title=f"Days of {self.sc.year}"),
            yaxis=dict(title=f"Time steps of a day ({self.sc.freq_unit})"),
            margin=dict(b=5, l=5, r=5, t=None if title else 5),
        )

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        data = timeseries.values.reshape((self.sc.steps_per_day, -1), order="F")[:, :]
        idx = self.sc.dated(timeseries).index
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

        if ent_name is not None:
            ser = self.sc.get_entity(ent=ent_name)
            colorbar_label = f"{ent_name} ({self.sc.get_unit(ent_name)})"
            title = f"{ent_name}: {self.sc.get_doc(ent_name)} ({self.sc.get_unit(ent_name)})"

        elif isinstance(timeseries, (np.ndarray, pd.Series)):
            ser = timeseries
            if isinstance(ser, np.ndarray):
                ser = pd.Series(ser)
        else:
            raise Exception("No timeseries specified!")

        data = ser.values.reshape((self.sc.steps_per_day, -1), order="F")[:, :]
        idx = self.sc.dated(ser).index

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
        fig.update_xaxes(title_text=f"Time ({self.sc.freq_unit})", row=1, col=1)
        fig.update_yaxes(title_text=f"Time ({self.sc.freq_unit})", row=2, col=1)
        fig.update_xaxes(title_text=f"Days of {self.sc.year}", row=2, col=1)

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
            colorbar_label = f"{colorbar_label} ({self.sc.get_unit(ent_name)})"
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

    def sankey(self, string_builder_func: Optional[Callable] = None, title: str = "") -> go.Figure:
        """Returns a Plotly sankey plot.

        Args:
            string_builder_func: Function that returns a string in the form of:
                type source target value
                F GAS CHP 1000
                E CHP EG 450
        """
        fig = self.get_sankey_fig(string_builder_func=string_builder_func, title=title)

        # self._plot_plotly_fig(fig, filename=self.sc._res_fp / f"{title}.html")
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

            if hp.get_dims(ent_name) == "T":
                data = self.sc.dated(data)

        return self._get_line_fig(data)

    def _get_line_fig(self, data: Union[pd.Series, pd.DataFrame]) -> go.Figure:

        # TODO: check dimensions and unstack if needed to cope with multidimensional data.

        df = _convert_to_df(data)

        plotly_data = []
        for _, ser in df.items():
            trace = go.Scatter(x=ser.index, y=ser.values)
            plotly_data.append(trace)

        return go.Figure(data=plotly_data)

    def violin_T(
        self, etypes: Tuple = ("P", "dQ", "dH"), unit: str = "kW", show_zero: bool = False
    ):
        df = self.sc.get_flat_T_df(lambda n: hp.get_etype(n) in etypes)

        if not show_zero:
            df = df.loc[:, (df != 0).any(axis=0)]

        df = df.sort_index(axis=1)
        y_scaler = 0.24 * len(df.columns)
        fig, ax = plt.subplots(1, figsize=(12, 0.4 + y_scaler))
        sns.violinplot(
            data=df,
            orient="h",
            scale="width",
            color="lightblue",
            ax=ax,
            cut=0,
            linewidth=1,
            width=0.95,
        )
        ax.margins(x=0)
        ax.set_xlabel(unit)
        hp.add_thousands_formatter(ax, y=False)
        sns.despine()

    def line_T(
        self,
        flatten_to_T: bool = True,
        etypes: Tuple = ("P", "dQ"),
        what: str = "v",
        steps: bool = True,
        tickformat: Optional[str] = "%a, %d.\n %b %Y",
        show_zero: bool = False,
        df=None,
        dated: bool = True,
    ) -> go.Figure:
        """Returns a Plotly line plot of all entities with the dimension T using Plotly express.

        Attention: the figure might be very big.

        Args:
            what: `p`for parameters and `v` or `r` for variables.
            dated: If index has datetimes.
        """
        if df is None:
            if flatten_to_T:
                df = self.sc.get_flat_T_df(lambda n: hp.get_etype(n) in etypes)
            else:
                df = self.sc.get_var_par_dic(what)["T"]

        if dated:
            df = self.sc.dated(df)

        if not show_zero:
            df.loc[:, (df != 0).any(axis=0)]

        ser = df.stack()
        ser.index = ser.index.rename(["T", "ent"])
        df = ser.rename("values").reset_index()

        def get_etype_desc(ent_name: str):
            etype = hp.get_etype(ent_name)
            try:
                etype_obj = getattr(Etypes, etype, None)
                desc = getattr(etype_obj, "en", None)
                unit = getattr(etype_obj, "units", " ")[0]
                if desc is not None:
                    return f"{desc} ({unit})"
            except AttributeError:
                return ""

        df["etype"] = df.ent.apply(lambda n: f"{hp.get_etype(n)}<br>{get_etype_desc(n)}").astype(
            "category"
        )
        df["desc"] = df.ent.apply(get_etype_desc).astype("category")
        df["doc"] = df.ent.apply(lambda n: self.sc.get_doc(n.split("[")[0])).astype("category")
        # df["weekday"] = df["T"].apply(lambda date: date.weekday_name).astype("category")
        # df["comp"] = df.ent.apply(lambda x: hp.get_component(x)).astype("category")
        df["ent"] = df["ent"].astype("category")

        y_scaler = len(df.ent.unique())
        fig = px.line(
            df,
            x="T",
            y="values",
            color="ent",
            facet_row="etype",
            # color_discrete_sequence=px.colors.qualitative.Light24,
            height=max(300, 27 * y_scaler),
            hover_data=["desc", "doc"],
        )
        if steps:
            fig.update_traces(line=dict(width=1.5, shape="hv"))  # shape=hv makes step function
        fig.update_traces(line_width=1.5)
        fig.update_yaxes(title="", matches=None)
        if tickformat is not None:
            fig.update_xaxes(
                tickformat=tickformat, dtick=24 * 60 * 60 * 1000
            )  # dtick in milliseconds
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
        """EXPERIMENTAL.

        Returns Plotly line plots of all desired entities with Matplotlib.

        Args:
            which_ents: A dictionary which dimensions as keys and as values, desired entities to plot.
            which_dims: Specify which dimensions to plot. e.g. 'TS'
            t_start: Start time step
            t_end: End time step
            what: `p`for parameters and `v` or `r` for variables.
            **pltargs: e.g.: drawstyle = 'steps-post' or kind = 'bar'

        Example:
            ```
            sc.plot.plot(which_ents={'T': ['E_BES_T'], 'TS': ['G_PS_TS']}, linestyle='steps')
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
                    data = self.sc.dated(dims_dic[dims][which_ents[dims]], y_dtindex)
                    data.plot(title=title_dim, **pltargs)
                else:
                    data = self.sc.dated(
                        dims_dic[dims][which_ents[dims]][:][t_start:t_end], y_dtindex
                    )
                    data.plot(title=title_dim, **pltargs)

            elif len(dims) == 2:
                for ent in which_ents[dims]:
                    title_ent = f"{self.sc.id}: {ent} ({dim_str})"
                    if t_end is None:
                        data = self.sc.dated(dims_dic[dims][ent].unstack(), y_dtindex)
                        data.plot(title=title_ent, **pltargs)
                    else:
                        data = self.sc.dated(
                            dims_dic[dims][ent][t_start:t_end].unstack(), y_dtindex
                        )
                        data.plot(title=title_ent, **pltargs)

            elif len(dims) >= 3:
                for ent in which_ents[dims]:
                    title_ent = f"{self.sc}: {ent} ({dim_str})"
                    if t_end is None:
                        data = self.sc.dated(dims_dic[dims][ent].unstack().unstack(), y_dtindex)
                        data.plot(title=title_ent, **pltargs)
                    else:
                        data = self.sc.dated(
                            dims_dic[dims][ent][t_start:t_end].unstack().unstack(), y_dtindex
                        )
                        data.plot(title=title_ent, **pltargs)

    def _get_all_plottable_entities_from(self, dims_dic) -> Dict[str, str]:
        return {dims: list(dims_dic[dims].columns) for dims in dims_dic if dims != ""}

    def _get_sankey_df(self, string_builder_func: Optional[Callable] = None) -> pd.DataFrame:
        s = (
            self.sc.make_sankey_string_from_collectors()
            if string_builder_func is None
            else string_builder_func(self.sc)
        )
        df = pd.read_csv(StringIO(textwrap.dedent(s)), sep=" ")
        df["value"] /= 1e3
        return df

    def get_sankey_fig(
        self, string_builder_func: Optional[Callable] = None, title: str = ""
    ) -> go.Figure:
        """Returns Plotly Sankey figure.

        Args:
            string_builder_func: Function that returns a string in the form of:
                type source target value
                F GAS CHP 1000
                E CHP EG 450
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
            arrangement="freeform",
            orientation="h",
            valueformat=".2f",
            valuesuffix="MWh",
            node=dict(
                pad=10, thickness=10, line=dict(width=0), label=label, color="hsla(0, 0%, 0%, 0.5)"
            ),
            link=dict(source=source, target=target, value=value, color=link_color),
        )

        layout = dict(title=title, font=dict(size=15), margin=dict(t=35, b=5, l=5, r=5))

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        return go.Figure(data=[data], layout=layout)

    def merit_order(self, **kwargs) -> Tuple["fig", "ax", "ax_right"]:
        return plots.merit_order(year=self.sc.year, country=self.sc.country, **kwargs)

    def describe(
        self,
        filter_str: str = "",
        sort_values: bool = False,
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
        entTypeDic = self._get_filled_entTypeDic(
            include_pars=include_pars, include_vars=include_vars and hasattr(self.sc, "res")
        )

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
                    orig_df=data, sort_values=sort_values, filter_str=filter_str
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
        steps_per_day = self.sc.steps_per_day

        assert len(series) % steps_per_day == 0, (
            "Timeseries doesn't fit the steps per day. There are "
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
            interpolation="None",
            aspect="auto",
            **imshow_kws,
        )

        ax.set(yticks=hours)
        ax.set_ylabel(f"Time ({self.sc.freq_unit})")
        ax.set_xlabel("Time (Days)", labelpad=10)
        ax.xaxis_date()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        date_format = mdates.DateFormatter("%b")
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(direction="out")

        return fig, ax, img

    def ts_balance(
        self,
        data: Dict[str, List],
        data_ylabel: str,
        data_conversion_factor: float,
        addon_ts: str,
        addon_ts_ylabel: str,
        addon_conversion_factor: float,
        colors: Dict[str, str],
        ts_slicer: Union[str, slice] = slice(None),
    ) -> go.Figure:
        """Plot time series as stacked balance.

        Args:
            data: A dictionary with the keys `pos` and `neg`. The values can be strings or
            Tuples with a string and a pandas series.
            addon_ts: An additional time series e.g. the electricity price.
            colors: Colors that refer to the strings in `data`.
            ts_slicer: A time slicer, e.g. `2019-04-30` to pick only one day.
        """

        sc = self.sc

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        for direction in data:
            for x in data[direction]:
                ent_ser = sc.get_entity(x) if isinstance(x, str) else x[1]
                ent_name = x if isinstance(x, str) else x[0]
                component = hp.get_component(x) if isinstance(x, str) else x[0]
                ser = sc.dated(ent_ser)[ts_slicer]
                values = ser.values if direction == "pos" else -ser.values
                fig.add_trace(
                    go.Scatter(
                        x=ser.index.tolist(),
                        y=values * data_conversion_factor,
                        legendgroup=direction,
                        line=dict(shape="hv", width=0),
                        fillcolor=colors.get(component, "red"),
                        mode="lines",
                        name=ent_name,
                        showlegend=True,
                        stackgroup=direction,
                    ),
                    row=1,
                    col=1,
                )

        ser = sc.dated(sc.get_entity(addon_ts))[ts_slicer]

        fig.add_trace(
            go.Scatter(
                x=ser.index.tolist(),
                y=ser.values * addon_conversion_factor,
                line=dict(color="black", shape="hv", width=1),
                mode="lines",
                name=addon_ts,
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            template="plotly_white",
            legend=dict(traceorder="grouped+reversed", tracegroupgap=50, yanchor="top", y=0.89),
            yaxis=dict(title=data_ylabel, domain=[0.28, 1.0]),
            yaxis2=dict(title=addon_ts_ylabel, domain=[0.0, 0.25]),
        )
        return fig


def _make_diverging_norm(ser, imshow_kws) -> Colormap:
    colors = ["white", "indianred", "darkred"]

    if ser.min() < 0 < ser.max():
        imshow_kws.update(norm=TwoSlopeNorm(vmin=ser.min(), vcenter=0.0, vmax=ser.max()))
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
