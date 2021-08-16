import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from draf import helper as hp
from draf.tsa.peak_load import PeakLoadAnalyzer


class DemandAnalyzer:
    def __init__(self, p_el: pd.Series, year: int = 2020, freq: str = "15min") -> None:
        self.p_el = p_el
        self.freq = freq
        self.year = year

    def dated(self, data: pd.Series):
        data = data.copy()
        data.index = hp.make_datetimeindex(year=self.year, freq=self.freq)
        return data

    def show_stats(self):
        self.line_plot()
        self.line_plot_ordered()
        self._make_violinplot()
        self._print_stats()

    def analyze_peaks(self, target_quantile: int = 0.95):
        pla = PeakLoadAnalyzer(self.p_el.values, figsize=(10, 2))
        self.pla.histo(peak_reduction=self.p_el.max() - self.p_el.quantile(target_quantile))

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
        ax = sns.violinplot(y=self.p_el, cut=0, width=0.5, scale="width", color="lightblue", ax=ax)
        ax.set_ylabel("$P_{el}$ [kW]")
        ax.set_xlim()
        ax.set_ylim(bottom=0)
        ax.set_title("Annotated violin plot")
        hp.add_thousands_formatter(ax, x=False)
        ax.get_xaxis().set_visible(False)
        sns.despine(bottom=True)

        datas = [
            ("Max", self.p_el.max(), "right"),
            ("95 percentile", self.p_el.quantile(q=0.95), "left"),
            ("75 percentile", self.p_el.quantile(q=0.75), "left"),
            ("Mean", self.p_el.mean(), "left"),
            ("Median", self.p_el.median(), "right"),
            ("25 percentile", self.p_el.quantile(q=0.25), "left"),
            ("Min", self.p_el.min(), "right"),
        ]

        for what, value_string, align in datas:
            flipper = -1 if align == "right" else 1
            ax.text(
                x=0.3 * flipper,
                y=value_string,
                s=f"{what}: {value_string:,.0f}",
                color="k",
                ha=align,
                va="center",
            )
            ax.annotate(
                "",
                (0.3 * flipper, value_string),
                (0, value_string),
                arrowprops=dict(arrowstyle="-", linestyle="--", alpha=0.3),
            )

    def _print_stats(self):
        step_width = hp.get_step_width(self.freq)
        sum_value, sum_unit = hp.auto_fmt(self.p_el.sum() * step_width, "kWh")
        data = [
            ("Number of data points", f"{len(self.p_el):,.0f}", ""),
            ("Peak-to-average-ratio:", f"{self.p_el.max()/self.p_el.mean():,.2f}", ""),
            ("Annual sum:", f"{sum_value:,.2f}", sum_unit),
        ]

        col_width = [max([len(word) for word in col]) for col in zip(*data)]

        for row in data:
            print(
                (row[0]).rjust(col_width[0]), row[1].rjust(col_width[1]), row[2].ljust(col_width[2])
            )
