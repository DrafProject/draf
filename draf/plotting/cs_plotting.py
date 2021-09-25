import logging
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns
from ipywidgets import interact
from pandas.io.formats.style import Styler as pdStyler

from draf import helper as hp
from draf.plotting.base_plotter import BasePlotter
from draf.plotting.scen_plotting import COLORS, ScenPlotter

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)

NAN_REPRESENTATION = "-"


class CsPlotter(BasePlotter):
    """Plotter for case studies.

    Args:
        cs: The CaseStudy object, containing all scenarios.
    """

    def __init__(self, cs: "CaseStudy"):
        self.figsize = (16, 4)
        self.cs = cs
        self.notebook_mode: bool = self.script_type() == "jupyter"
        self.optimize_layout_for_reveal_slides = False

    def __getstate__(self):
        """For serialization with pickle."""
        return None

    def yields(self, gradient: bool = False) -> pdStyler:
        """Returns a styled pandas table with cost and carbon savings, and avoidance cost.

        Args:
            gradient: If the table is coloured with a gradient.
        """
        cs = self.cs

        savings = cs.pareto.iloc[0] - cs.pareto

        rel_savings = savings / cs.pareto.iloc[0]

        avoid_cost = -savings["C_TOT_"] / savings["CE_TOT_"] * 1e6  # in k€/kgCO2eq  # in €/tCO2eq

        df = pd.DataFrame(
            {
                ("_____Absolute_____", "Costs"): cs.pareto["C_TOT_"],
                ("_____Absolute_____", "Emissions"): cs.pareto["CE_TOT_"] / 1e3,
                ("_Abolute savings_", "Costs"): savings["C_TOT_"],
                ("_Abolute savings_", "Emissions"): savings["CE_TOT_"] / 1e3,
                ("_Relative savings_", "Costs"): rel_savings["C_TOT_"],
                ("_Relative savings_", "Emissions"): rel_savings["CE_TOT_"],
                ("", "Emission avoidance costs"): avoid_cost,
                ("", "CAPEX"): pd.Series(cs.get_ent("C_TOT_inv_")),
                ("", "OPEX"): pd.Series(cs.get_ent("C_TOT_op_")),
            }
        )
        df[("", "Payback time")] = df[("", "CAPEX")] / (df[("", "OPEX")].iloc[0] - df[("", "OPEX")])

        def color_negative_red(val):
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        df = df.fillna(0)

        if gradient:
            styled_df = (
                df.style.background_gradient(subset=["_____Absolute_____"], cmap="Greens")
                .background_gradient(subset=["_Abolute savings_"], cmap="Reds")
                .background_gradient(subset=["_Relative savings_"], cmap="Blues")
                .background_gradient(subset=[""], cmap="Greys_r")
            )

        else:
            styled_df = df.style.applymap(color_negative_red)

        styled_df = styled_df.format(
            {
                ("_____Absolute_____", "Costs"): "{:,.0f} k€",
                ("_____Absolute_____", "Emissions"): "{:,.0f} t",
                ("_Abolute savings_", "Costs"): "{:,.0f} k€",
                ("_Abolute savings_", "Emissions"): "{:,.0f} t",
                ("_Relative savings_", "Costs"): "{:,.2%}",
                ("_Relative savings_", "Emissions"): "{:,.2%}",
                ("", "Emission avoidance costs"): "{:,.0f} €/t",
                ("", "CAPEX"): "{:,.0f} k€",
                ("", "OPEX"): "{:,.0f} k€",
                ("", "Payback time"): "{:,.1f} a",
            }
        )

        return styled_df

    def pareto(
        self,
        use_plotly: bool = True,
        target_c_unit: Optional[str] = None,
        target_ce_unit: Optional[str] = None,
        c_dict: Dict = None,
        label_verbosity: int = 1,
        do_title: bool = True,
    ) -> go.Figure:
        """Plots the Pareto points in an scatter plot.

        Args:
            use_plotly: If True, Plotly is used, else Matplotlib
            target_c_unit: The unit of the cost.
            target_ce_unit: The unit of the carbon emissions.
            c_dict: colors the Pareto points according to key-strings in their scenario doc
                e.g. {"FLAT": "green", "TOU": "blue", "RTP": "red"}
            label_verbosity: Choose between 1: "id", 2: "name", 3: "doc".
            do_title: If title is shown.
        """
        cs = self.cs
        pareto = cs.pareto.copy()
        scens_list = cs.scens_list

        options = {1: "id", 2: "name", 3: "doc"}
        pareto.index = [getattr(sc, options[label_verbosity]) for sc in scens_list]

        units = dict()
        for x, target_unit in zip(["C_TOT_", "CE_TOT_"], [target_c_unit, target_ce_unit]):
            pareto[x], units[x] = hp.auto_fmt(
                pareto[x], scens_list[0].get_unit(x), target_unit=target_unit
            )

        def get_title(pareto: pd.DataFrame) -> str:
            ce_saving = pareto.iloc[0, 1] - pareto.iloc[:, 1].min()
            ce_saving_rel = 100 * (pareto.iloc[0, 1] - pareto.iloc[:, 1].min()) / pareto.iloc[0, 1]
            c_saving = pareto.iloc[0, 0] - pareto.iloc[:, 0].min()
            c_saving_rel = 100 * (pareto.iloc[0, 0] - pareto.iloc[:, 0].min()) / pareto.iloc[0, 0]

            title = (
                f"Max. cost savings: {c_saving:,.1f} {units['C_TOT_']} ({c_saving_rel:.2f}%)\n"
                f"Max. carbon savings: {ce_saving:.1f} {units['CE_TOT_']} ({ce_saving_rel:.2f}%)"
            )
            if use_plotly:
                title = title.replace("\n", "<br>")
            return title

        def get_colors(c_dict: Dict) -> List:
            return [c for sc in scens_list for i, c in c_dict.items() if i in sc.doc]

        colors = "black" if c_dict is None else get_colors(c_dict)
        ylabel = f"Costs [{units['C_TOT_']}]"
        xlabel = f"Carbon emissions [{units['CE_TOT_']}]"

        if use_plotly:
            hwr_ = "<b>id:</b> {}<br><b>name:</b> {}<br><b>doc:</b> {}<br>"

            trace = go.Scatter(
                x=pareto["CE_TOT_"],
                y=pareto["C_TOT_"],
                mode="markers+text",
                text=list(pareto.index) if label_verbosity else None,
                hovertext=[hwr_.format(sc.id, sc.name, sc.doc) for sc in scens_list],
                textposition="bottom center",
                marker=dict(size=12, color=colors, showscale=False),
            )
            data = [trace]
            layout = go.Layout(
                hovermode="closest",
                title=get_title(pareto) if do_title else "",
                xaxis=dict(title=xlabel),
                yaxis=dict(title=ylabel),
                margin=dict(l=5, r=5, b=5),
            )

            if self.optimize_layout_for_reveal_slides:
                layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

            fig = go.Figure(data=data, layout=layout)
            return fig

        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            pareto.plot.scatter("CE_TOT_", "C_TOT_", s=30, marker="o", ax=ax, color=colors)
            ax.set(ylabel=ylabel, xlabel=xlabel)
            if do_title:
                ax.set(title=get_title(pareto))

            for sc_name in list(pareto.index):
                ax.annotate(
                    s=sc_name,
                    xy=(pareto["CE_TOT_"][sc_name], pareto["C_TOT_"][sc_name]),
                    rotation=45,
                    ha="left",
                    va="bottom",
                )
            return fig

    def pareto_curves(
        self,
        groups: List[str] = None,
        c_unit: Optional[str] = None,
        ce_unit: Optional[str] = None,
        c_dict: Optional[Dict] = None,
        label_verbosity: int = 0,
        do_title: bool = True,
    ) -> go.Figure:
        """EXPERIMENTAL: Plot based on pareto() considering multiple pareto curve-groups."""

        def get_hover_text(sc, ref_scen):
            sav_C = ref_scen.res.C_TOT_ - sc.res.C_TOT_
            sav_C_fmted, unit_C = hp.auto_fmt(sav_C, sc.get_unit("C_TOT_"))
            sav_C_rel = sav_C / ref_scen.res.C_TOT_
            sav_CE = ref_scen.res.CE_TOT_ - sc.res.CE_TOT_
            sav_CE_fmted, unit_CE = hp.auto_fmt(sav_CE, sc.get_unit("CE_TOT_"))
            sav_CE_rel = sav_CE / ref_scen.res.CE_TOT_

            return "<br>".join(
                [
                    f"<b>Id:</b> {sc.id}",
                    f"<b>Name:</b> {sc.name}",
                    f"<b>Doc:</b> {sc.doc}",
                    f"<b>Cost savings:</b> {sav_C_fmted:.2f} {unit_C} ({sav_C_rel:.3%})",
                    f"<b>Emission savings:</b> {sav_CE_fmted:.2f} {unit_CE} ({sav_CE_rel:.3%})",
                ]
            )

        def get_text(sc, label_verbosity) -> str:
            if label_verbosity == 1:
                return sc.id
            elif label_verbosity == 2:
                return sc.name
            elif label_verbosity == 3:
                return sc.doc
            elif label_verbosity == 4:
                return f"α={sc.params.k_PTO_alpha_:.2f}"

        cs = self.cs
        pareto = cs.pareto.copy()
        scens = cs.scens_list

        colors = [
            "#606269",  # some grey for REF
            # "#F26535",  # for FLAT
            # "#FCC706",   # for TOU
            # "#9BAF65",   # for RTP
            "#1f77b4",  # (plotly default) muted blue
            "#ff7f0e",  # (plotly default) safety orange
            "#2ca02c",  # (plotly default) cooked asparagus green
            "#d62728",  # (plotly default) brick red
            "#9467bd",  # (plotly default) muted purple
            "#8c564b",  # (plotly default) chestnut brown
            "#e377c2",  # (plotly default) raspberry yogurt pink
            "#7f7f7f",  # (plotly default) middle gray
            "#bcbd22",  # (plotly default) curry yellow-green
            "#17becf",  # (plotly default) blue-teal
        ]

        if isinstance(groups, list) and c_dict is None:
            c_dict = {g: colors[i] for i, g in enumerate(groups)}

        if c_dict is None and groups is None:
            c_dict = {"": "black"}
            c_dict = dict(REF="#606269", FLAT="#F26535", TOU="#FCC706", RTP="#9BAF65")

        if pareto.empty:
            logger.warning("\nPareto-Dataframe is empty!")
            return

        pareto["C_TOT_"], c_unit = hp.auto_fmt(
            pareto["C_TOT_"], scens[0].get_unit("C_TOT_"), target_unit=c_unit
        )
        pareto["CE_TOT_"], ce_unit = hp.auto_fmt(
            pareto["CE_TOT_"], scens[0].get_unit("CE_TOT_"), target_unit=ce_unit
        )
        title = ""

        layout = go.Layout(
            hovermode="closest",
            title=title if do_title else "",
            xaxis=dict(title=f"Carbon emissions [{ce_unit}]"),
            yaxis=dict(title=f"Costs [{c_unit}]"),
        )

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        data = []
        for ix, c in c_dict.items():
            scens_ = [sc for sc in scens if ix in sc.name]
            pareto_ = [getattr(cs.scens, ix) for ix in pareto.index]

            trace = go.Scatter(
                x=[pareto.loc[sc.id, "CE_TOT_"] for sc in scens_],
                y=[pareto.loc[sc.id, "C_TOT_"] for sc in scens_],
                mode="lines+markers+text" if bool(label_verbosity) else "lines+markers",
                text=[get_text(sc, label_verbosity) for sc in scens_]
                if bool(label_verbosity)
                else None,
                hovertext=[get_hover_text(sc, ref_scen=cs.REF_scen) for sc in scens_],
                textposition="bottom center",
                marker=dict(size=12, color=c, showscale=False),
                name=ix,
            )
            data.append(trace)

        fig = go.Figure(layout=layout, data=data)

        if not self.notebook_mode:
            fp = str(cs._res_fp / "plotly_pareto_scatter.html")
            py.offline.plot(fig, filename=fp)

        return fig

    def heatmap_interact(
        self,
        what: str = "p",
        dim: str = "T",
        select: Tuple[Union[int, str]] = None,
        cmap: str = None,
        show_info: bool = True,
    ) -> go.Figure:
        """Returns an interactive heatmap widget that enables browsing through time series.

        Args:
            what: Selects between Variables ('v') and Parameters ('p').
            dim: Dimensions to filter.
            select: Tuple of indexers for data with additional dimension(s) besides the time.
            cmap: Color scale.
            show_info: If additional information such as Scenario, Entity, Stats are displayed.
        """
        cs = self.cs
        sc = cs.any_scen
        layout = go.Layout(
            title=None,
            xaxis=dict(title=f"Days of year {cs.year}"),
            yaxis=dict(title="Time of day"),
            margin=dict(b=5, l=5, r=5),
        )
        fig = go.FigureWidget(layout=layout)
        heatmap = fig.add_heatmap(colorscale=cmap)

        @interact(scen_id=cs.scens_ids, ent=sc.get_var_par_dic(what)[dim].keys())
        def update(scen_id, ent):
            with fig.batch_update():
                sc = getattr(cs.scens, scen_id)
                ser = sc.get_var_par_dic(what)[dim][ent]
                title_addon_if_select = ""

                if len(dim) > 1:
                    if select is None:
                        ser = ser.sum(level=0)
                    else:
                        indexer = select if isinstance(select, Tuple) else (select,)
                        ser = ser.loc[(slice(None, None),) + indexer]
                        s = ", ".join([f"{k}={v}" for k, v in zip(dim[1:], indexer)])
                        title_addon_if_select = f"[{s}]"

                data = ser.values.reshape((cs.steps_per_day, -1), order="F")[:, :]
                idx = cs.dated(ser).index
                heatmap.data[0].x = pd.date_range(start=idx[0], end=idx[-1], freq="D")
                heatmap.data[0].y = pd.date_range(
                    start="0:00", freq=cs.freq, periods=cs.steps_per_day
                )
                heatmap.data[0].z = data
                heatmap.layout.yaxis.tickformat = "%H:%M"
                if show_info:
                    unit = "-" if sc.get_unit(ent) == "" else sc.get_unit(ent)
                    heatmap.layout.title = (
                        "<span style='font-size:medium;'>"
                        f"{grey('Scenario:')} <b>{scen_id}</b> ◦ {sc.doc}"
                        f"<br>{grey(' ⤷ Entity:')} <b>{ent}</b>{title_addon_if_select} ◦ {sc.get_doc(ent)}"
                        f"<br>{grey('    ⤷ Stats:')} ∑ <b>{data.sum():,.2f}</b> ◦ Ø <b>{data.mean():,.2f}</b>"
                        f" ◦ min <b>{data.min():,.2f}</b> ◦ max <b>{data.max():,.2f}</b>"
                        f"  [<b>{unit}</b>]"
                        "</span>"
                    )

        return fig

    def sankey_interact(self, string_builder_func: Callable) -> go.Figure:
        """Returns an interactive Sankey plot widget to browse scenarios.

        Args:
            string_builder_func: Function that returns a space-seperated table with
                the columns type, source, targe, value. e.g.
                ```
                type source target value
                F GAS CHP 1000
                E CHP EG 450
                ```
        """
        cs = self.cs

        data = dict(
            type="sankey",
            node=dict(
                pad=10,
                thickness=10,
                line=dict(color="white", width=0),
                color="hsla(0, 0%, 0%, 0.5)",
            ),
        )

        layout = dict(title=None, font=dict(size=14), margin=dict(t=5, b=5, l=5, r=5))

        fig = go.FigureWidget(data=[data], layout=layout)
        sankey = fig.add_sankey()

        sankeys_dic = {}

        for scen_name, sc in cs.valid_scens.items():
            df = sc.plot._get_sankey_df(string_builder_func)
            source_s, target_s, value = (list(df[s]) for s in ["source", "target", "value"])

            label = list(set(source_s + target_s))
            source = [label.index(x) for x in source_s]
            target = [label.index(x) for x in target_s]

            link_color = [COLORS[x] for x in df["type"].values.tolist()]

            sankeys_dic[scen_name] = dict(
                source=source, target=target, value=value, color=link_color
            )

        @interact(scen_name=cs.valid_scens.keys())
        def update(scen_name):
            with fig.batch_update():
                sankey["data"][0]["link"] = sankeys_dic[scen_name]
                sankey["data"][0]["node"].label = label
                sankey["data"][0]["node"].color = "hsla(0, 0%, 0%, 0.5)"
                sankey["data"][0].orientation = "h"
                sankey["data"][0].valueformat = ".2f"
                sankey["data"][0].valuesuffix = "MWh"

        return fig

    def big_plot(self, string_builder_func, sc: "Scenario" = None, sort: bool = True) -> go.Figure:
        """Experimental: Builds a big plot containing other subplots.

        Args:
            string_builder_func: Function that returns a space-seperated table with
                the columns type, source, targe, value. e.g.
                ```
                type source target value
                F GAS CHP 1000
                E CHP EG 450
                ```
            sc: Scenario object, which is selected.
            sort: If scenarios are sorted by total costs.

        """
        cs = self.cs
        sc = cs.REF_scen if sc is None else sc
        r = sc.res
        p = sc.params

        if not hasattr(cs.REF_scen.res, "C_TOT_op_") or not hasattr(cs.REF_scen.res, "C_TOT_inv_"):
            for scen in cs.scens_list:
                scen.res.C_TOT_op_ = scen.res.C_TOT_
                scen.res.C_TOT_inv_ = 0

        css = cs.ordered_valid_scens if sort else cs.valid_scens

        def get_table_trace():
            trace = go.Table(
                header=dict(
                    values=["Total", f"{r.C_TOT_:,.0f}", "€ / a"],
                    line=dict(color="lightgray"),
                    align=["left", "right", "left"],
                    font=dict(size=12),
                ),
                cells=dict(
                    values=[
                        ["Operation", "Invest", "Savings", "Depreciation", "Peakload"],
                        [
                            f"{r.C_TOT_op_:,.0f}",
                            f"{r.C_TOT_inv_:,.0f}",
                            f"{cs.REF_scen.res.C_TOT_ - r.C_TOT_:,.0f}",
                            f"{r.C_TOT_inv_ * p.k__AF_:,.0f}",
                            f"{r.P_EG_buyPeak_:,.0f}",
                        ],
                        ["k€/a", "k€", "k€/a", "k€/a", "kW"],
                    ],
                    line=dict(color="lightgray"),
                    align=["left", "right", "left"],
                    font=dict(size=12),
                ),
                domain=dict(x=[0, 0.45], y=[0.4, 1]),
            )
            return trace

        def get_bar_traces():
            d = OrderedDict(
                {
                    sc.id: [sc.res.C_TOT_inv_ * sc.params.k__AF_, sc.res.C_TOT_op_]
                    for sc in css.values()
                }
            )
            df = pd.DataFrame(d, ["Depreciation", "Operation"])

            def get_opacity(id: str) -> float:
                if id == sc.id:
                    return 1.0
                elif id == "REF":
                    return 1.0
                else:
                    return 0.3

            traces = [
                go.Bar(
                    x=df.columns,
                    y=df.loc[ent, :],
                    name=ent,
                    marker=dict(
                        color=["blue", "#de7400"][i], opacity=[get_opacity(id) for id in df.columns]
                    ),
                )
                for i, ent in enumerate(df.index)
            ]

            return traces

        sankey_trace = sc.plot.get_sankey_fig(string_builder_func)["data"][0]
        sankey_trace.update(dict(domain=dict(x=[0.5, 1], y=[0, 1])))

        data = [sankey_trace, get_table_trace()] + get_bar_traces()

        layout = dict(
            title=f"Scenario {sc.id}: {sc.name} ({sc.doc})",
            font=dict(size=12),
            barmode="stack",
            xaxis=dict(domain=[0, 0.45]),
            yaxis=dict(domain=[0, 0.4]),
            legend=dict(x=0, y=0.5),
            margin=dict(t=30, b=5, l=5, r=5),
        )

        if self.optimize_layout_for_reveal_slides:
            layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

        return go.Figure(data=data, layout=layout)

    def table(
        self,
        what: str = "p",
        show_unit: bool = True,
        show_doc: bool = True,
        show_src: bool = False,
        show_etype: bool = False,
        show_comp: bool = False,
        show_desc: bool = False,
        show_dims: bool = False,
    ) -> pdStyler:
        """Creates a table with all scalars.

        Args:
            what: (p)arameters or (v)ariables.
            show_unit: If units are shown.
            show_doc: If entity docs are shown.
        """
        cs = self.cs

        tmp_list = [
            pd.Series(sc.get_var_par_dic(what)[""], name=name) for name, sc in cs.scens_dic.items()
        ]

        df = pd.concat(tmp_list, axis=1)
        if show_unit:
            df["Unit"] = [cs.any_scen.get_unit(ent_name) for ent_name in df.index]
        if show_etype:
            df["Etype"] = [hp.get_etype(ent_name) for ent_name in df.index]
        if show_comp:
            df["Comp"] = [hp.get_component(ent_name) for ent_name in df.index]
        if show_desc:
            df["Desc"] = [hp.get_desc(ent_name) for ent_name in df.index]
        if show_doc:
            df["Doc"] = [cs.any_scen.get_doc(ent_name) for ent_name in df.index]
        if show_src:
            df["Src"] = [cs.any_scen.get_src(ent_name) for ent_name in df.index]
        df.index.name = what
        cm = sns.light_palette("green", n_colors=20, as_cmap=True)

        def highlight_diff1(s):
            other_than_REF = s == df.iloc[:, 0]
            return ["color: lightgray" if v else "" for v in other_than_REF]

        def highlight_diff2(s):
            other_than_REF = s != df.iloc[:, 0]
            return ["font-weight: bold" if v else "" for v in other_than_REF]

        left_aligner = list(df.dtypes[df.dtypes == object].index)
        return (
            df.style.background_gradient(cmap=cm)
            .apply(highlight_diff1, subset=df.columns[1:])
            .apply(highlight_diff2, subset=df.columns[1:])
            .set_caption("Note: Bold font indicate deviation from first/reference scenario.")
            .set_properties(subset=left_aligner, **{"text-align": "left"})
            .set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        )

    @hp.copy_doc(ScenPlotter.describe, start="Args:")
    def describe(self, **kwargs) -> None:
        """Prints a description of all Parameters and Results for all scenarios."""
        for sc in self.cs.scens_list:
            sc.plot.describe(**kwargs)

    def describe_interact(self):
        cs = self.cs

        def f(sc):
            cs.scens.get(sc).plot.describe()

        interact(f, sc=cs.scens_ids)

    def times(self, yscale: str = "linear", stacked: bool = True) -> None:
        """Barplot of the calculation times (Params, Vars, Model, Solve).

        Args:
            yscale: 'log' makes the y-axis logarithmic. Default: 'linear'.
            stacked: If bars are stacked.
        """
        cs = self.cs
        df = pd.DataFrame(
            {
                "Params": pd.Series(cs.get_ent("t__params_")),
                "Vars": pd.Series(cs.get_ent("t__vars_")),
                "Model": pd.Series(cs.get_ent("t__model_")),
                "Solve": pd.Series(cs.get_ent("t__solve_")),
            }
        )

        total_time = df.sum().sum()
        fig, ax = plt.subplots(figsize=(12, 3))
        df.plot.bar(
            stacked=stacked,
            ax=ax,
            title=f"Total time: {total_time:,.0f} s (≈ {total_time/60:,.0f} minutes)",
        )
        ax.set(ylabel="Wall time [s]", yscale=yscale)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False)
        sns.despine()

    def invest(self, include_capx: bool = True) -> go.Figure:
        """Annotated heatmap of existing and new capacities and a barchart of according CAPEX
         and OPEX.

        Args:
            include_capx: If existing capacities are included.

        """
        cs = self.cs

        df = self._get_capa_for_all_scens(which="CAPn")
        df = df.rename(columns=lambda x: f"<b>{x}</b>")

        if include_capx:
            df2 = self._get_capa_for_all_scens(which="CAPx")
            df = pd.concat([df2, df], sort=False, axis=1)

        fig = _get_capa_heatmap(df)

        ser = pd.Series(cs.get_ent("C_TOT_inv_"))
        fig.add_trace(
            go.Bar(y=ser.index.tolist(), x=ser.values, xaxis="x2", yaxis="y2", orientation="h")
        )

        ser = pd.Series(cs.get_ent("C_TOT_op_"))
        fig.add_trace(
            go.Bar(y=ser.index.tolist(), x=ser.values, xaxis="x3", yaxis="y3", orientation="h")
        )

        unit = cs.REF_scen.get_unit("C_TOT_inv_")
        capx_adder = " (decision variables in <b>bold</b>)" if include_capx else ""
        fig.update_layout(
            margin=dict(t=5, l=5, r=5, b=5),
            xaxis=dict(domain=[0, 0.78], title=f"Capacity [kW or kWh] of component{capx_adder}"),
            xaxis2=dict(domain=[0.80, 0.90], anchor="y2", title=f"CAPEX [{unit}]", side="top"),
            yaxis2=dict(anchor="x2", showticklabels=False),
            xaxis3=dict(domain=[0.92, 1], anchor="y3", title=f"OPEX [{unit}]", side="top"),
            yaxis3=dict(anchor="x3", showticklabels=False),
            showlegend=False,
        )
        fig.update_yaxes(matches="y")
        return fig

    def _get_capa_for_all_scens(self, which: str) -> pd.DataFrame:
        """'which' can be 'CAPn' or 'CAPx'"""
        cs = self.cs
        return pd.DataFrame(
            {n: sc.get_CAP(which=which, agg=True) for n, sc in cs.scens_dic.items()}
        ).T

    def _get_correlation(self, ent1: str, ent2: str) -> pd.Series:
        """EXPERIMENTAL: Returns correlation coefficients between two entities for all scenarios."""
        d = dict()
        cs = self.cs
        for sc in cs.scens_list:
            ser1 = sc.get_entity(ent1)
            ser2 = sc.get_entity(ent2)
            d[sc.id] = ser1.sum(level=0).corr(ser2.sum(level=0))
        return pd.Series(d)


def _get_capa_heatmap(df) -> go.Figure:
    data = df.where(df > 0)
    fig = ff.create_annotated_heatmap(
        data.values,
        x=data.columns.tolist(),
        y=data.index.tolist(),
        annotation_text=data.applymap(float_to_int_to_string).values,
        showscale=False,
        colorscale="OrRd",
        font_colors=["white", "black"],
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, side="top"),
        yaxis=dict(showgrid=False, autorange="reversed", title="Scenario"),
        width=900,
        height=400,
    )
    set_font_size(fig=fig, size=9)
    make_high_values_white(fig=fig, data=data)
    return fig


def make_high_values_white(fig, data, diverging: bool = False) -> None:
    if diverging:
        data = data.abs()
    minz = data.min().min()
    maxz = data.max().max()
    threshold = (minz + maxz) / 2
    for i in range(len(fig.layout.annotations)):
        ann_text = fig.layout.annotations[i].text
        if ann_text != NAN_REPRESENTATION:
            f = float(ann_text.replace(",", ""))
            if diverging:
                f = abs(f)
            if f > threshold:
                fig.layout.annotations[i].font.color = "white"


def set_font_size(fig, size: int = 9) -> None:
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = size


def grey(s: str):
    return f"<span style='font-size:small;color:grey;font-family:monospace;'>{s}</span>"


def float_to_int_to_string(afloat):
    return f"{afloat:,.0f}".replace("nan", NAN_REPRESENTATION)


def float_to_string_with_precision_1(afloat):
    return f"{afloat:.1f}".replace("nan", NAN_REPRESENTATION)


def float_to_string_with_precision_2(afloat):
    return f"{afloat:.2f}".replace("nan", NAN_REPRESENTATION)
