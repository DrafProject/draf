import itertools
import logging
import math
import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns
from IPython import display
from IPython.core.display import HTML
from ipywidgets import interact, widgets
from pandas.api.types import is_numeric_dtype
from pandas.io.formats.style import Styler as pdStyler

from draf import helper as hp
from draf.plotting.base_plotter import BasePlotter
from draf.plotting.plotting_util import make_clickable_src
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

    def __call__(self):
        funcs = {"Tables": self.tables, "Plots": self.plots}

        @interact(what=widgets.ToggleButtons(options=funcs.keys(), description=" "))
        def f(what):
            return funcs[what]()

    def plots(self):
        funcs = {
            "Pareto": self.pareto,
            "Sankey": self.sankey,
            "Heatmap": self.heatmap_all_T,
            "Collectors": self.collectors,
            "Violin_T": self.violin_T,
            "Times": self.times,
            "Annual el.": self.collector_balance,
        }

        @interact(what=widgets.ToggleButtons(options=funcs.keys(), description=" "))
        def f(what):
            return funcs[what]()

    def tables(self):
        cs = self.cs
        funcs = {
            "Yields ": ("yields_table", " fa-eur fa-lg"),
            "Capacities ": ("capa_table", " fa-cubes fa-lg"),
            "Investments ": ("invest_table", "money fa-lg"),
            "TES capa ": ("capa_TES_table", " fa-cubes fa-lg"),
            "Parameters ": ("p_table", "table fa-lg"),
            "Variables ": ("v_table", "table fa-lg"),
            "eGrid ": ("eGrid_table", " fa-plug fa-lg"),
            "eFlex ": ("eFlex_table", " fa-balance-scale"),
            "BES ": ("bes_table", "fa-solid fa-battery-half fa-lg"),
            "Calc. time ": ("time_table", " fa-clock-o fa-lg"),
            "Collectors ": ("collector_table", " fa fa-bus"),
            "Correlations ": ("correlation_table", " fa fa-connectdevelop"),
        }

        ui = widgets.ToggleButtons(
            options={k: v[0] for k, v in funcs.items()},
            description=" ",
            icons=[v[1] for v in funcs.values()],
            # for icons see https://fontawesome.com/v4.7/icons/
        )

        @interact(table=ui, gradient=True, caption=True)
        def f(table, gradient, caption):
            kw = dict()
            if table in ("p_table", "v_table"):
                what, func = table.split("_")
                kw.update(what=what)
            else:
                func = table
            kw.update(gradient=gradient, caption=caption)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    display(getattr(cs.plot, func)(**kw))
            except (AttributeError, KeyError) as e:
                display(HTML("<h2>⚠️ No data</h2>"))
                print(e)

    def pareto_table(self, gradient: bool = False, caption: bool = False) -> pdStyler:
        cs = self.cs
        df = cs.pareto
        s = df.style.set_table_styles(get_leftAlignedIndex_style()).format(
            {v: "{:,.0f} " + f"{cs.REF_scen.get_unit(v)}" for v in cs.obj_vars}
        )

        if gradient:
            s = s.background_gradient(cmap="OrRd")
        if caption:
            s = s.set_caption("Pareto table")
        return s

    def _get_internal_rates_of_return(self, base_years: int = 15) -> pd.Series:
        cs = self.cs
        c_inv = cs.get_ent("C_TOT_inv_")
        c_op = cs.get_diff("C_TOT_op_")
        data = [npf.irr([-inv] + base_years * [op]) for inv, op in zip(c_inv.values, c_op.values)]
        return pd.Series(data, c_inv.index)

    @staticmethod
    def _get_discounted_payback_period(rate: float, payback_period: pd.Series) -> pd.Series:
        data = np.log(1 / (1 - (rate * payback_period.values))) / np.log(1 + rate)
        # https://github.com/epri-dev/StorageVET/blob/576a01dec9effa174c944c56a88f748da5072bff/storagevet/Finances.py#L610
        return pd.Series(data, payback_period.index)

    def yields_table(
        self, gradient: bool = False, nyears_for_irr: int = 15, caption: bool = False
    ) -> pdStyler:
        """Returns a styled pandas table with cost and carbon savings, and avoidance cost."""
        cs = self.cs
        df = pd.DataFrame(
            {
                ("Total annualized", "Costs"): cs.get_ent("C_TOT_"),
                ("Total annualized", "Emissions"): cs.get_ent("CE_TOT_") / 1e3,
                ("Absolute savings", "Costs"): cs.get_diff("C_TOT_"),
                ("Absolute savings", "Emissions"): cs.get_diff("CE_TOT_") / 1e3,
                ("Relative savings", "Costs"): cs.get_diff("C_TOT_")
                / cs.REF_scen.get_entity("C_TOT_"),
                ("Relative savings", "Emissions"): cs.get_diff("CE_TOT_")
                / cs.REF_scen.get_entity("CE_TOT_"),
                ("", "C_inv"): cs.get_ent("C_TOT_inv_"),
                ("", "CapEx"): cs.get_ent("C_TOT_invAnn_"),
                ("", "OpEx"): cs.get_ent("C_TOT_op_"),
                ("", "EAC"): -cs.get_diff("C_TOT_") / cs.get_diff("CE_TOT_") * 1e6,
                ("", "PP"): cs.get_ent("C_TOT_inv_")
                / ((cs.get_diff("C_TOT_op_") + cs.get_diff("C_TOT_RMI_"))).replace(
                    np.inf, np.nan  # infinity is not supported by background gradient
                ),
            }
        )

        df[("", "DPP")] = self._get_discounted_payback_period(
            rate=cs.REF_scen.params.k__r_, payback_period=df[("", "PP")]
        )
        df[("", "IRR")] = self._get_internal_rates_of_return(nyears_for_irr)

        def color_negative_red(val):
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        if gradient:
            s = df.style.background_gradient(cmap="OrRd")
        else:
            s = df.style.applymap(color_negative_red)

        s = s.format(
            {
                ("Total annualized", "Costs"): "{:,.0f} k€",
                ("Total annualized", "Emissions"): "{:,.0f} t",
                ("Absolute savings", "Costs"): "{:,.0f} k€",
                ("Absolute savings", "Emissions"): "{:,.0f} t",
                ("Relative savings", "Costs"): "{:,.2%}",
                ("Relative savings", "Emissions"): "{:,.2%}",
                ("", "C_inv"): "{:,.0f} k€",
                ("", "CapEx"): "{:,.0f} k€/a",
                ("", "OpEx"): "{:,.0f} k€/a",
                ("", "EAC"): "{:,.0f} €/t",
                ("", "PP"): "{:,.1f} a",
                ("", "DPP"): "{:,.1f} a",
                ("", "IRR"): "{:,.1%}",
            }
        ).set_table_styles(get_leftAlignedIndex_style() + get_multiColumnHeader_style(df))
        if caption:
            s = s.set_caption(
                "<u>Legend</u>: "
                "<b>C_inv</b>: Investment Costs, "
                "<b>CapEx</b>: Capital Expenditures (annualized investment costs), "
                "<b>OpEx</b>: Operating Expeses, "
                "<b>EAC</b>: Emissions Avoidance Costs, "
                "<b>PP</b>: Payback Period, "
                "<b>DPP</b>: Discounted Payback Period, "
                f"<b>IRR</b>: Internal Rate of Return for {nyears_for_irr} years"
                "</br>"
                f"<u>Note</u>: <b>{cs.REF_scen.params.k__r_:.1%}</b> discount rate assumed."
            )
        return s

    def bes_table(self, gradient: bool = False, caption: bool = False) -> pdStyler:
        data = [
            ("CAPn", "{:,.0f} kWh", lambda df, cs: cs.get_ent("E_BES_CAPn_")),
            (
                "W_out",
                "{:,.0f} MWh/a",
                lambda df, cs: [sc.gte(sc.get_ent("P_BES_out_T")) / 1e3 for sc in cs.scens],
            ),
            ("Charging_cycles", "{:,.0f}", lambda df, cs: df["W_out"] / (df["CAPn"] / 1e3)),
        ]
        return self.base_table(data, gradient, caption, caption_text="BES table")

    def eGrid_table(
        self, gradient: bool = False, pv: bool = False, caption: bool = False
    ) -> pdStyler:
        data = [
            ("P_max", "{:,.0f} kW", lambda df, cs: cs.get_ent("P_EG_buyPeak_")),
            ("P_max_reduction", "{:,.0f} kW", lambda df, cs: cs.get_diff("P_EG_buyPeak_")),
            (
                "P_max_reduction_rel",
                "{:,.1%}",
                lambda df, cs: df["P_max_reduction"] / df["P_max"].iloc[0],
            ),
            ("t_use", "{:,.0f} h", lambda df, cs: [sc.get_EG_full_load_hours() for sc in cs.scens]),
            (
                "t_use_diff_rel",
                "{:,.1%}",
                lambda df, cs: (df["t_use"] - df["t_use"].iloc[0]) / df["t_use"].iloc[0],
            ),
            (
                "W_buy",
                "{:,.2f} GWh/a",
                lambda df, cs: [sc.gte(sc.res.P_EG_buy_T) / 1e6 for sc in cs.scens],
            ),
            (
                "W_sell",
                "{:,.2f} GWh/a",
                lambda df, cs: [sc.gte(sc.res.P_EG_sell_T) / 1e6 for sc in cs.scens],
            ),
        ]

        if pv:
            data.append(
                (
                    "W_pv_own",
                    "{:,.2f} MWh/a",
                    lambda df, cs: [sc.gte(sc.res.P_PV_OC_T) / 1e3 for sc in cs.scens],
                )
            )

        caption_text = (
            "<u>Legend</u>: "
            "<b>P_max</b>: Peak load, "
            "<b>P_max_redu</b>: Peak load reduction compared to REF, "
            "<b>t_use</b>: Full load hours, "
            "<b>W_buy / W_sell</b>: Bought / sold electricity."
        )

        return self.base_table(data, gradient, caption, caption_text=caption_text)

    def eFlex_table(self, gradient: bool = False, caption: bool = True) -> pdStyler:
        data = [
            (
                "W_buy",
                "{:,.2f} GWh/a",
                lambda df, cs: [sc.res.P_EG_buy_T.sum() / 1e6 for sc in cs.scens],
            ),
            (
                "EWAP_buy",
                "{:,.0f} €/MWh",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T * sc.params.c_EG_T).sum() / sc.res.P_EG_buy_T.sum() * 1e3
                    for sc in cs.scens
                ],
            ),
            (
                "EWACEF_buy",
                "{:,.2f} t/MWh",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T * sc.params.ce_EG_T).sum() / sc.res.P_EG_buy_T.sum()
                    for sc in cs.scens
                ],
            ),
            (
                "ECER_buy",
                "{:,.0f} €/t",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T * sc.params.c_EG_T).sum()
                    * 1e3
                    / (sc.res.P_EG_buy_T * sc.params.ce_EG_T).sum()
                    for sc in cs.scens
                ],
            ),
            (
                "W_net",
                "{:,.2f} GWh/a",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T.sum() - sc.res.P_EG_sell_T.sum()) / 1e6 for sc in cs.scens
                ],
            ),
            (
                "EWAP_net",
                "{:,.0f} €/MWh",
                lambda df, cs: [
                    (
                        (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T)
                        / (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T).sum()
                        * sc.params.c_EG_T
                    ).sum()
                    * 1e3
                    for sc in cs.scens
                ],
            ),
            (
                "EWACEF_net",
                "{:,.2f} t/MWh",
                lambda df, cs: [
                    (
                        (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T)
                        / (sc.res.P_EG_buy_T - sc.res.P_EG_sell_T).sum()
                        * sc.params.ce_EG_T
                    ).sum()
                    for sc in cs.scens
                ],
            ),
            (
                "W_sell",
                "{:,.2f} GWh/a",
                lambda df, cs: [sc.res.P_EG_sell_T.sum() / 1e6 for sc in cs.scens],
            ),
            (
                "EWAP_sell",
                "{:,.0f} €/MWh",
                lambda df, cs: [
                    ((sc.res.P_EG_sell_T) / (sc.res.P_EG_sell_T).sum() * sc.params.c_EG_T).sum()
                    * 1e3
                    for sc in cs.scens
                ],
            ),
            (
                "EWACEF_sell",
                "{:,.2f} t/MWh",
                lambda df, cs: [
                    ((sc.res.P_EG_sell_T) / sc.res.P_EG_sell_T.sum() * sc.params.ce_EG_T).sum()
                    for sc in cs.scens
                ],
            ),
            (
                "avg P_devAbs",
                "{:,.0f} kW",
                lambda df, cs: [
                    ((sc.res.P_EG_buy_T - cs.REF_scen.res.P_EG_buy_T).abs()).mean()
                    for sc in cs.scens
                ],
            ),
            (
                "avg P_devAbs (%)",
                "{:,.1%}",
                lambda df, cs: df["avg P_devAbs"] / cs.REF_scen.res.P_EG_buy_T.mean(),
            ),
            (
                "corr (P,c_RTP)",
                "{:,.3f}",
                lambda df, cs: [sc.res.P_EG_buy_T.corr(sc.params.c_EG_RTP_T) for sc in cs.scens],
            ),
            (
                "corr (P_dev,c_RTP)",
                "{:,.3f}",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T - cs.REF_scen.res.P_EG_buy_T).corr(sc.params.c_EG_RTP_T)
                    for sc in cs.scens
                ],
            ),
            (
                "abs_flex_score",
                "{:,.1f}",
                lambda df, cs: df["avg P_devAbs"] * -df["corr (P_dev,c_RTP)"],
            ),
            (
                "rel_flex_score",
                "{:,.1%}",
                lambda df, cs: df["avg P_devAbs (%)"] * -df["corr (P_dev,c_RTP)"],
            ),
            (
                "corr (P,ce_EG)",
                "{:,.3f}",
                lambda df, cs: [sc.res.P_EG_buy_T.corr(sc.params.ce_EG_T) for sc in cs.scens],
            ),
            (
                "corr (P_dev,ce_EG)",
                "{:,.3f}",
                lambda df, cs: [
                    (sc.res.P_EG_buy_T - cs.REF_scen.res.P_EG_buy_T).corr(sc.params.ce_EG_T)
                    for sc in cs.scens
                ],
            ),
        ]
        caption_text = (
            "<u>Legend</u>: <b>W_buy/net/sell</b>: Purchased/net/sold annual electricity,"
            " <b>EWAP/EWACEF_buy/net/sell</b>: Energy-weighted average price/carbon emission factor"
            " of purchased/net/sold electricity, <b>ECER_buy</b>: energy-based cost-emission ratio,"
            " <b>avg P_devAbs</b>: mean absolute deviation of the scenarios' purchased power and"
            " the one of the reference case study (baseline), <b>corr (P_dev,c_RTP)</b>: Pearson"
            " correlation coefficient between purchase difference to the baseline and the real"
            " time prices.<b>abs_flex_score</b> (<b>rel_flex_score</b>): Column 1 (2) multiplied"
            " with negated values of column 3."
        )

        return self.base_table(data, gradient, caption, caption_text)

    def base_table(
        self,
        data: List[Tuple[str, str, Callable]],
        gradient: bool = False,
        caption: bool = False,
        caption_text: Optional[str] = None,
    ):
        cs = self.cs
        df = pd.DataFrame(index=cs.scens_ids)

        for name, fmt, func in data:
            df[name] = func(df, cs)

        format_dict = {name: fmt for name, fmt, func in data}

        s = df.style.format(format_dict).set_table_styles(get_leftAlignedIndex_style())

        if gradient:
            s = s.background_gradient(cmap="OrRd")
        if caption:
            s = s.set_caption(caption_text)
        return s

    def pareto(
        self,
        use_plotly: bool = True,
        target_c_unit: Optional[str] = None,
        target_ce_unit: Optional[str] = None,
        c_dict: Dict = None,
        label_verbosity: int = 1,
        do_title: bool = True,
        start_x_with_0=True,
        start_y_with_0=True,
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
            start_x_with_0, start_y_with_0: Set zero as lower bound for the x or y axis.
        """
        # workaround to prevent "Loading [MathJax]/extensions/MathMenu.js":
        # see https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
        py.io.kaleido.scope.mathjax = None

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

        def get_colors(c_dict: Dict) -> List:
            return [c for sc in scens_list for i, c in c_dict.items() if i in sc.doc]

        colors = "black" if c_dict is None else get_colors(c_dict)
        ylabel = f"Annualized costs ({units['C_TOT_']})"
        xlabel = f"Carbon emissions ({units['CE_TOT_']})"

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
                title=dict(
                    text=get_pareto_title(pareto, units)
                    .replace("\n", "<br>")
                    .replace("CO2eq", "CO<sub>2</sub>eq")
                    if do_title
                    else "",
                    font_size=15,
                ),
                xaxis=dict(title=xlabel),
                yaxis=dict(title=ylabel),
                margin=dict(l=5, r=5, b=5, t=35),
            )

            if self.optimize_layout_for_reveal_slides:
                layout = hp.optimize_plotly_layout_for_reveal_slides(layout)

            fig = go.Figure(data=data, layout=layout)
            if start_y_with_0:
                ymax = fig.data[0].y.max() * 1.1
                fig.update_yaxes(range=[0, ymax])
            if start_x_with_0:
                xmax = fig.data[0].x.max() * 1.1
                fig.update_xaxes(range=[0, xmax])
            return fig

        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            pareto.plot.scatter("CE_TOT_", "C_TOT_", s=30, marker="o", ax=ax, color=colors)
            ax.set(ylabel=ylabel, xlabel=xlabel)
            if do_title:
                ax.set(title=get_pareto_title(pareto, units))
            for sc_name in list(pareto.index):
                ax.annotate(
                    sc_name,
                    xy=(pareto["CE_TOT_"][sc_name], pareto["C_TOT_"][sc_name]),
                    rotation=45,
                    ha="left",
                    va="bottom",
                )

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
            xaxis=dict(title=f"Carbon emissions ({ce_unit})"),
            yaxis=dict(title=f"Costs ({c_unit})"),
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
        cmap: str = "OrRd",
        show_info: bool = True,
    ) -> go.Figure:
        """Returns an interactive heatmap widget that enables browsing through time series.

        Args:
            what: `p`for parameters and `v` or `r` for variables.
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
                sc = cs.scens.get(scen_id)
                ser = sc.get_var_par_dic(what)[dim][ent]
                title_addon_if_select = ""

                if len(dim) > 1:
                    if select is None:
                        ser = ser.groupby(level=0).sum()
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
                    heatmap.layout.title = self._get_heatmap_info_title(
                        sc, ent, title_addon_if_select, data
                    )

        return fig

    def _get_heatmap_info_title(self, sc, ent, title_addon_if_select, data):
        base_ent = ent.split("[")[0]
        unit = "-" if sc.get_unit(base_ent) == "" else sc.get_unit(base_ent)
        scen_id = sc.id
        return (
            f"<span style='font-size:medium;'>{grey('Scenario:')} <b>{scen_id}</b> ◦"
            f" {sc.doc}<br>{grey(' ⤷ Entity:')} <b>{ent}</b>{title_addon_if_select} ◦"
            f" {sc.get_doc(base_ent)}<br>{grey('    ⤷ Stats:')} ∑ <b>{data.sum():,.2f}</b> ◦"
            f" Ø <b>{data.mean():,.2f}</b> ◦ min <b>{data.min():,.2f}</b> ◦ max"
            f" <b>{data.max():,.2f}</b>  (<b>{unit}</b>)</span>"
        )

    def heatmap_all_T(self, cmap: str = "OrRd", show_info: bool = True) -> go.Figure:
        """Returns an interactive heatmap widget that enables browsing through time series.

        Args:
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
        data_dic = {sc.id: sc.get_flat_T_df() for sc in cs.scens}
        entity_list = sorted(data_dic[cs.any_scen.id].columns)
        ts = widgets.Dropdown(options=entity_list, description="time series")

        @interact(scenario=data_dic.keys(), ts=ts)
        def update(scenario, ts):
            with fig.batch_update():
                ser = data_dic[scenario][ts]
                data = ser.values.reshape((cs.steps_per_day, -1), order="F")[:, :]
                idx = cs.dated(ser).index
                heatmap.data[0].x = pd.date_range(start=idx[0], end=idx[-1], freq="D")
                heatmap.data[0].y = pd.date_range(
                    start="0:00", freq=cs.freq, periods=cs.steps_per_day
                )
                heatmap.data[0].z = data
                heatmap.layout.yaxis.tickformat = "%H:%M"
                if show_info:
                    heatmap.layout.title = self._get_heatmap_info_title(
                        sc=sc, ent=ts, title_addon_if_select="", data=data
                    )

        return fig

    def collectors(self) -> go.Figure:
        cs = self.cs

        @interact(scenario=cs.scens_ids, filter_etype="P", auto_convert_units=False)
        def f(scenario, filter_etype, auto_convert_units):
            sc = cs.scens.get(scenario)
            display(
                sc.plot.collectors(filter_etype=filter_etype, auto_convert_units=auto_convert_units)
            )

    def heatmap(self):
        cs = self.cs

        @interact(
            what=widgets.ToggleButtons(options=["Parameters", "Variables"], description="show")
        )
        def f(what):
            plot = cs.plot.heatmap_interact(what={"Parameters": "p", "Variables": "r"}[what])
            display(plot)

    def sankey(self) -> go.Figure:
        cs = self.cs

        @interact(scenario=cs.scens_ids)
        def f(scenario):
            sc = cs.scens.get(scenario)
            display(sc.plot.sankey())

    def sankey_interact(self, string_builder_func: Optional[Callable] = None) -> go.Figure:
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

    def collector_balance(
        self,
        sink="P_EL_sink_T",
        source="P_EL_source_T",
        xlabel="Electricity consumption (GWh/a)",
        factor=1e-6,
        nlabel_rows=2,
    ):
        cs = self.cs

        sinks = cs.get_collector_values(sink)
        if source is not None:
            sources = cs.get_collector_values(source)
            df = pd.concat([sinks, -sources])
        else:
            df = sinks
        df = df * factor
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.5, lw=0.5)
        df.T.plot.barh(stacked=True, ax=ax, cmap="tab20_r", width=0.7)
        ax.set_xlabel(xlabel)
        # plt.yticks(rotation=20, ha="right")

        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        # sort legend
        handles, labels = ax.get_legend_handles_labels()
        if source is not None:
            nsinks = len(sinks.index)
            nsources = len(sources.index)
            order = list(range(nsinks))[::-1] + list(range(nsinks, nsources + nsinks))
            order = order[::-1]
            handles = [handles[idx] for idx in order]
            labels = [labels[idx] for idx in order]
        else:
            handles, labels = handles[::-1], labels[::-1]

        # remove unused labels without changing the colours
        dismissed_labels = df[df.sum(1) == 0].index
        h_and_ls = [(h, l) for h, l in zip(handles, labels) if l not in dismissed_labels]
        handles, labels = zip(*h_and_ls)

        ax.invert_yaxis()

        ncol = math.ceil(len(handles) / nlabel_rows)
        ax.legend(
            flip(handles, ncol),
            flip(labels, ncol),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
            ncol=ncol,
            columnspacing=0.8,
            handletextpad=0.2,
            handlelength=0.8,
        )

        sns.despine()

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
            for scen in cs.scens:
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

    def ref_table(self, verbose: bool = False):
        return self.table(
            show_src=True,
            show_comp=verbose,
            show_desc=verbose,
            show_etype=verbose,
            show_dims=verbose,
            clickable_urls=True,
            only_ref=True,
        )

    def table(
        self,
        what: str = "p",
        only_scalars: bool = False,
        show_unit: bool = True,
        show_doc: bool = True,
        show_src: bool = False,
        show_etype: bool = False,
        show_comp: bool = False,
        show_desc: bool = False,
        show_dims: bool = False,
        gradient: bool = False,
        filter_func: Optional[Callable] = None,
        only_ref: bool = False,
        number_format: str = "{:n}",
        caption: bool = False,
        clickable_urls: bool = False,
        sort_by_name: bool = False,
    ) -> pdStyler:
        """Creates a table with all entities.
        For multi-dimensional entities, the mean is displayed.

        Args:
            what: `p`for parameters and `v` or `r` for variables.
            show_unit: If units are shown.
            show_doc: If entity docs are shown.
            number_format: How numerical data are formatted. e.g. `"{:n}"` or `"{:,.0f}"`
        """
        # FIXME: prevent clickable_urls from switching off highlight_diff
        cs = self.cs
        d = {cs.REF_scen.id: cs.REF_scen} if only_ref else cs.scens_dic

        if only_scalars:
            tmp_list = [
                pd.Series(sc.get_var_par_dic(what)[""], name=name) for name, sc in d.items()
            ]
        else:
            tmp_dic = dict(p="params", r="res", v="res")
            tmp_list = [
                pd.Series(
                    {k: hp.get_mean(i) for k, i in getattr(sc, tmp_dic[what]).get_all().items()},
                    name=name,
                )
                for name, sc in d.items()
            ]

        df = pd.concat(tmp_list, axis=1)

        if filter_func is not None:
            df = df.loc[df.index.map(filter_func)]

        sc = cs.any_scen
        if show_unit:
            df["Unit"] = [sc.get_unit(ent_name) for ent_name in df.index]
        if show_etype:
            df["Etype"] = [hp.get_etype(ent_name) for ent_name in df.index]
        if show_comp:
            df["Comp"] = [hp.get_component(ent_name) for ent_name in df.index]
        if show_desc:
            df["Desc"] = [hp.get_desc(ent_name) for ent_name in df.index]
        if show_dims:
            df["Dims"] = [hp.get_dims(ent_name) for ent_name in df.index]
        if show_doc:
            df["Doc"] = [sc.get_doc(ent_name) for ent_name in df.index]
        if show_src:
            df["Src"] = [sc.get_src(ent_name) for ent_name in df.index]
            if clickable_urls:
                df["Src"] = df["Src"].apply(make_clickable_src)
        df.index.name = what

        def highlight_diff1(s):
            other_than_REF = s == df.iloc[:, 0]
            return ["color: lightgray" if v else "" for v in other_than_REF]

        def highlight_diff2(s):
            other_than_REF = s != df.iloc[:, 0]
            return ["font-weight: bold" if v else "" for v in other_than_REF]

        if sort_by_name:
            df = df.sort_index()

        left_aligner = list(df.dtypes[df.dtypes == object].index)
        s = (
            df.style.format({n: number_format if is_numeric_dtype(df[n]) else "{:s}" for n in df})
            .apply(highlight_diff1, subset=df.columns[1:])
            .apply(highlight_diff2, subset=df.columns[1:])
            .set_properties(subset=left_aligner, **{"text-align": "left"})
            .set_table_styles([dict(selector="th", props=[("text-align", "left")])])
            .set_sticky(axis=1)
        )
        if caption:
            s = s.set_caption(
                "<b>Bold</b> numbers indicate deviation from"
                " <code>REF</code>-scenario. For multi-dimensional entities, the mean is displayed."
            )
        if gradient:
            s = s.background_gradient(cmap="OrRd", axis=1)
        if clickable_urls:
            return HTML(s.data.to_html(render_links=True, escape=False))
        else:
            return s

    def collector_table(self, gradient: bool = True, caption: bool = False):
        cs = self.cs
        df = pd.DataFrame(
            {k: pd.DataFrame(sc.collector_values).T.stack() for k, sc in cs.scens_dic.items()}
        )

        def highlight_diff1(s):
            other_than_REF = s == df.iloc[:, 0]
            return ["color: lightgray" if v else "" for v in other_than_REF]

        def highlight_diff2(s):
            other_than_REF = s != df.iloc[:, 0]
            return ["font-weight: bold" if v else "" for v in other_than_REF]

        df.index.names = ["Collector", "Component"]
        s = df.style.format("{:.3n}").apply(highlight_diff1).apply(highlight_diff2)
        if gradient:
            s = s.background_gradient(cmap="OrRd", axis=1)
        if caption:
            s = s.set_caption("Collector table")
        s = s.set_sticky(axis=1)
        return s

    def correlation_table(self, gradient, caption):
        @interact(scenario=self.cs.scens_ids)
        def f(scenario):
            sc = self.cs.scens.get(scenario)
            return sc.plot.correlation_table(gradient=gradient, caption=caption)

    @hp.copy_doc(ScenPlotter.describe, start="Args:")
    def describe(self, **kwargs) -> None:
        """Prints a description of all Parameters and Results for all scenarios."""
        for sc in self.cs.scens:
            sc.plot.describe(**kwargs)

    def describe_interact(self, **kwargs):
        @interact(scenario=self.cs.scens_ids)
        def f(scenario):
            sc = self.cs.scens.get(scenario)
            sc.plot.describe(**kwargs)

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

    def time_table(
        self, gradient: bool = False, caption: bool = False, number_format="{:.3n} s"
    ) -> pdStyler:
        cs = self.cs
        df = pd.DataFrame(
            {
                "Params": pd.Series(cs.get_ent("t__params_")),
                "Vars": pd.Series(cs.get_ent("t__vars_")),
                "Model": pd.Series(cs.get_ent("t__model_")),
                "Solve": pd.Series(cs.get_ent("t__solve_")),
            }
        )
        s = df.style.format(number_format).set_table_styles(get_leftAlignedIndex_style())
        if caption:
            s = s.set_caption("Calculation time (seconds)")
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        return s

    def invest_table(self, gradient: bool = False, caption: bool = False) -> pdStyler:
        cs = self.cs
        l = dict(
            C_TOT_inv_="Investment costs (k€)",
            C_TOT_invAnn_="CapEx (=annualized investment costs) (k€/a)",
        )
        df = pd.DataFrame(
            {
                desc: pd.DataFrame(
                    {n: sc.collector_values[which] for n, sc in cs.scens_dic.items()}
                )
                .sort_index(axis=0)
                .stack()
                for which, desc in l.items()
            }
        ).unstack(0)

        s = df.style.format("{:,.0f}").set_table_styles(
            get_leftAlignedIndex_style() + get_multiColumnHeader_style(df)
        )
        if caption:
            s = s.set_caption("Investment costs and CapEx per scenario and component.")
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        return s

    def invest(self, annualized: bool = True) -> go.Figure:
        cs = self.cs
        if annualized:
            ent_name = "C_TOT_invAnn_"
            title = "Annualized Investment cost (k€)"
        else:
            ent_name = "C_TOT_inv_"
            title = "Investment cost (k€)"

        df = pd.DataFrame({n: sc.collector_values[ent_name] for n, sc in cs.scens_dic.items()}).T

        fig = _get_capa_heatmap(df)
        fig.update_layout(
            margin=dict(t=80, l=5, r=5, b=5),
            xaxis=dict(title=dict(text="Component", standoff=0)),
            title=dict(text=title),
        )
        return fig

    def capas(
        self,
        include_capx: bool = True,
        subplot_x_anchors: Tuple = (0.79, 0.91),
        c_inv: bool = False,
        c_op: bool = False,
    ) -> go.Figure:
        """Annotated heatmap of existing and new capacities and a barchart of according C_inv
         and C_op.

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

        if c_inv:
            ser = pd.Series(cs.get_ent("C_TOT_inv_"))
            unit1 = cs.REF_scen.get_unit("C_TOT_inv_")
            ser, unit1 = hp.auto_fmt(ser, unit1)
            fig.add_trace(
                go.Bar(
                    y=ser.index.tolist(),
                    x=ser.values,
                    xaxis="x2",
                    yaxis="y2",
                    orientation="h",
                    marker_color="grey",
                )
            )

        if c_op:
            ser = pd.Series(cs.get_ent("C_TOT_op_"))
            unit2 = cs.REF_scen.get_unit("C_TOT_op_")
            ser, unit2 = hp.auto_fmt(ser, unit2)
            fig.add_trace(
                go.Bar(
                    y=ser.index.tolist(),
                    x=ser.values,
                    xaxis="x3",
                    yaxis="y3",
                    orientation="h",
                    marker_color="grey",
                )
            )

        margin = 0.01
        domain1 = (0, subplot_x_anchors[0] - margin)

        capx_adder = " (decision variables in <b>bold</b>)" if include_capx else ""
        fig.update_layout(
            margin=dict(t=5, l=5, r=5, b=5),
            xaxis=dict(domain=domain1, title=f"Capacity of component (kW or kWh){capx_adder}"),
        )
        if c_inv:
            domain2 = (subplot_x_anchors[0] + margin, subplot_x_anchors[1] - margin)
            fig.update_layout(
                xaxis2=dict(domain=domain2, anchor="y2", title=f"C_inv ({unit1})", side="top"),
                yaxis2=dict(anchor="x2", showticklabels=False),
            )
        if c_op:
            domain3 = (subplot_x_anchors[1] + margin, 1)
            fig.update_layout(
                xaxis3=dict(domain=domain3, anchor="y3", title=f"C_op ({unit2})", side="top"),
                yaxis3=dict(anchor="x3", showticklabels=False),
                showlegend=False,
            )
        fig.update_yaxes(matches="y")
        return fig

    def capa_table(
        self, gradient: bool = False, caption: bool = False, show_zero_cols: bool = True
    ) -> pdStyler:
        df = (
            pd.DataFrame(
                {which: self._get_capa_for_all_scens(which).stack() for which in ["CAPx", "CAPn"]}
            )
            .unstack()
            .reindex(self.cs.scens_ids)
        )
        if not show_zero_cols:
            df = df.loc[:, (df != 0).any(axis=0)]
            df = df.dropna(axis=1)

        s = df.style.format(precision=0, thousands=",").set_table_styles(
            get_leftAlignedIndex_style() + get_multiColumnHeader_style(df)
        )
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        if caption:
            s = s.set_caption("Existing (CAPx) and new (CAPn) capacity")
        return s

    def capa_TES_table(self, gradient: bool = False, caption: bool = False) -> pdStyler:
        cs = self.cs
        energy = cs.get_ent("Q_TES_CAPn_L").T.stack()
        volume = energy.apply(hp.get_TES_volume)
        df = pd.DataFrame({"Energy (kWh)": energy, "Volume (m³)": volume}).unstack()

        s = df.style.format(precision=0, thousands=",").set_table_styles(
            get_leftAlignedIndex_style() + get_multiColumnHeader_style(df)
        )
        if gradient:
            s = s.background_gradient(cmap="OrRd")
        if caption:
            s = s.set_caption("New TES capacity for different temperature levels.")
        return s

    def _get_capa_for_all_scens(self, which: str) -> pd.DataFrame:
        """'which' can be 'CAPn' or 'CAPx'"""
        cs = self.cs
        return pd.DataFrame(
            {n: sc.get_CAP(which=which, agg=True) for n, sc in cs.scens_dic.items()}
        ).T

    def _get_correlations(self, ent1: str, ent2: str) -> pd.Series:
        """Returns correlation coefficients between two entities for all scenarios."""
        d = dict()
        cs = self.cs
        for sc in cs.scens:
            ser1 = sc.get_entity(ent1)
            ser2 = sc.get_entity(ent2)
            d[sc.id] = ser1.groupby(level=0).sum().corr(ser2.groupby(level=0).sum())
        return pd.Series(d)

    def correlations(self, ent1: str, ent2: str):
        """Plots the Pearson correlation coefficient of the the given entities for each scenario."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ser = self._get_correlations(ent1=ent1, ent2=ent2)
        ser.plot.barh(ax=ax, width=0.85)
        ax.invert_yaxis()
        ax.margins(0.15, 0.15)
        plt.bar_label(ax.containers[0], fmt="%.2f", padding=5, color="grey")
        ax.set_xlabel("Correlation coefficient")
        sns.despine()

    def violin_T(self):
        cs = self.cs

        @interact(scenario=cs.scens_ids, show_zero=True)
        def f(scenario, show_zero):
            sc = cs.scens.get(scenario)
            return sc.plot.violin_T(show_zero=show_zero)


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
        width=200 + len(df.columns) * 40,
        height=200 + len(df) * 5,
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


def get_divider_nums(df):
    ser = pd.Series(df.columns.codes[0]).diff()
    return ser[ser != 0].index.tolist()[1:]


def get_divider(column_loc):
    return {"selector": f".col{column_loc}", "props": [("border-left", "1px solid black")]}


def get_multiColumnHeader_style(df):
    cols = {"selector": "th.col_heading", "props": [("text-align", "center")]}
    return [cols] + [get_divider(i) for i in get_divider_nums(df)]


def get_leftAlignedIndex_style():
    return [dict(selector="th.row_heading", props=[("text-align", "left")])]


def get_pareto_title(pareto: pd.DataFrame, units) -> str:
    ce_saving = pareto.iloc[0, 1] - pareto.iloc[:, 1].min()
    ce_saving_rel = 100 * (pareto.iloc[0, 1] - pareto.iloc[:, 1].min()) / pareto.iloc[0, 1]
    c_saving = pareto.iloc[0, 0] - pareto.iloc[:, 0].min()
    c_saving_rel = 100 * (pareto.iloc[0, 0] - pareto.iloc[:, 0].min()) / pareto.iloc[0, 0]
    return (
        f"Savings: {c_saving:,.2f} {units['C_TOT_']} ({c_saving_rel:.0f}%), "
        f"{ce_saving:.2f} {units['CE_TOT_']} ({ce_saving_rel:.0f}%)"
    )
