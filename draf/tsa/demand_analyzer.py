import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns

from draf import helper as hp
from draf.core.datetime_handler import DateTimeHandler
from draf.core.scenario import Scenario
from draf.tsa.peak_load import PeakLoadAnalyzer


class DemandAnalyzer(DateTimeHandler):
    def __init__(self, p_el: pd.Series, year: int = 2020, freq: str = "15min") -> None:
        self.p_el = p_el
        self._set_dtindex(year=year, freq=freq)

    def show_stats(self):
        self.line_plot()
        self.line_plot_ordered()
        self._make_violinplot()
        self._print_stats()

    def get_peak_load_analyzer(self) -> PeakLoadAnalyzer:
        return PeakLoadAnalyzer(self.p_el, year=self.year, freq=self.freq, figsize=(10, 2))

    def line_plot(self):
        _, ax = plt.subplots(1, figsize=(10, 2))
        data = self.dated(self.p_el)
        data.plot(linewidth=0.6, ax=ax, color="darkgray")
        sns.despine()
        ax.set_title("Load curve")
        ax.set_ylabel("$P_{el}$ [kW]")
        hp.add_thousands_formatter(ax, x=False)
        ax.set_ylim(bottom=0)

    def line_plot_ordered(self):
        _, ax = plt.subplots(1, figsize=(10, 2))
        data = self.p_el.sort_values(ascending=False).reset_index(drop=True)
        data.plot(linewidth=1.5, ax=ax, color="darkgray")
        ax.set_title("Ordered annual duration curve")
        sns.despine()
        ax.set_ylabel("$P_{el}$ [kW]")
        hp.add_thousands_formatter(ax)
        ax.set_ylim(bottom=0)

    def _make_violinplot(self):
        _, ax = plt.subplots(figsize=(8, 4))
        ax = sns.violinplot(y=self.p_el, cut=0, width=0.4, scale="width", color="lightblue", ax=ax)
        ax.set_ylabel("$P_{el}$ [kW]")
        ax.set_xlim()
        ax.set_ylim(bottom=0)
        ax.set_title("Annotated violin plot")
        hp.add_thousands_formatter(ax, x=False)
        ax.get_xaxis().set_visible(False)
        sns.despine(bottom=True)

        self._annotate_violins_left_side(ax)
        self._annotate_violins_right_side(ax)

    def _annotate_violins_left_side(self, ax):
        to_annotate = [
            ("Max", self.p_el.max()),
            ("Mean", self.p_el.mean()),
            ("Min", self.p_el.min()),
        ]

        for what, value_string in to_annotate:
            ax.text(
                x=-0.25,
                y=value_string,
                s=f"{what}: {value_string:,.0f} kW",
                color="k",
                ha="right",
                va="center",
                fontweight="bold",
            )
            ax.annotate(
                "",
                (0, value_string),
                (-0.25, value_string),
                arrowprops=dict(arrowstyle="-", linestyle="--", alpha=1),
            )

    def _annotate_violins_right_side(self, ax):
        percentile_range = (70, 80, 90, 95, 97, 99)
        percentile_locs = np.linspace(0.1, 0.95, len(percentile_range))
        y_max = self.p_el.max()

        for pcnt, pcnt_loc in zip(percentile_range, percentile_locs):
            quantile = pcnt / 100
            value = self.p_el.quantile(q=quantile)
            edge_x = 0.21
            highlight = pcnt % 10 == 0
            alpha = 0.5 if highlight else 1
            ax.annotate(
                text="",
                xy=(0, value),
                xytext=(edge_x, value),
                arrowprops=dict(arrowstyle="-", linestyle="--", alpha=alpha, shrinkA=0),
            )
            ax.annotate(
                text="",
                xy=(edge_x, value),
                xytext=(0.31, pcnt_loc * y_max),
                arrowprops=dict(arrowstyle="-", linestyle="--", alpha=alpha, shrinkB=0),
            )
            ax.text(
                x=0.31,
                y=pcnt_loc * y_max,
                s=f"{pcnt} percentile: {value:,.0f} kW (= {y_max - value:,.0f} kW reduction)",
                ha="left",
                va="center",
                alpha=alpha,
            )

    def _print_stats(self):
        step_width = hp.get_step_width(self.freq)
        sum_value, sum_unit = hp.auto_fmt(self.p_el.sum() * step_width, "kWh")
        std_value, std_unit = hp.auto_fmt(self.p_el.std(), "kW")
        data = [
            ("Year:", f"{self.year}", ""),
            ("Frequency:", f"{hp.int_from_freq(self.freq)}", "Minutes"),
            ("Number of time steps:", f"{len(self.p_el):,.0f}", ""),
            ("Annual sum:", f"{sum_value:,.2f}", sum_unit),
            ("Standard deviation:", f"{std_value:,.2f}", std_unit),
            ("Peak-to-average ratio:", f"{self.p_el.max()/self.p_el.mean():,.2f}", ""),
        ]

        col_width = [max([len(word) for word in col]) for col in zip(*data)]

        for row in data:
            print(
                (row[0]).rjust(col_width[0]), row[1].rjust(col_width[1]), row[2].ljust(col_width[2])
            )

    def heatmap(self) -> go.Figure:
        sc = Scenario(year=self.year, freq=self.freq)
        return sc.plot.heatmap_py(self.p_el)
