import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from draf import helper as hp
from draf.core.datetime_handler import DateTimeHandler
from draf.tsa.peak_load import PeakLoadAnalyzer


class DemandAnalyzer(DateTimeHandler):
    def __init__(self, p_el: pd.Series, year: int = 2020, freq: str = "15min") -> None:
        self.p_el = p_el
        self._set_dtindex(year=year, freq=freq)

    def get_peak_load_analyzer(self) -> PeakLoadAnalyzer:
        return PeakLoadAnalyzer(self.p_el, year=self.year, freq=self.freq, figsize=(10, 2))

    def show_peaks(
        self, target_percentile: int = 95, c_GRID: float = 0.12, c_GRID_peak: float = 50.0
    ) -> PeakLoadAnalyzer:
        pla = PeakLoadAnalyzer(self.p_el, year=self.year, freq=self.freq, figsize=(10, 2))
        pla.set_prices(c_GRID=c_GRID, c_GRID_peak=c_GRID_peak)
        pla.histo(target_percentile)
        return pla

    def show_stats(self):
        self.print_stats()
        self.violin_plot()
        self.line_plot()
        self.ordered_line_plot()
        self.averages_plot()
        self.weekdays_plot()

    def print_stats(self) -> None:

        step_width = hp.get_step_width(self.freq)
        sum_value, sum_unit = hp.auto_fmt(self.p_el.sum() * step_width, "kWh")
        std_value, std_unit = hp.auto_fmt(self.p_el.std(), "kW")
        peak_to_average = self.p_el.max() / self.p_el.mean()

        data = [
            ("Year:", f"{self.year}", ""),
            ("Frequency:", f"{hp.int_from_freq(self.freq)}", "minutes"),
            ("Length:", f"{len(self.p_el):,.0f}", "time steps"),
            ("Annual sum:", f"{sum_value:,.2f}", sum_unit),
            ("Standard deviation:", f"{std_value:,.2f}", std_unit),
            ("Peak-to-average ratio:", f"{peak_to_average:,.2f}", ""),
            ("Full-load hours:", f"{8760*peak_to_average**-1:,.2f}", "h"),
        ]

        col_width = [max([len(word) for word in col]) for col in zip(*data)]
        for row in data:
            print(
                (row[0]).rjust(col_width[0]), row[1].rjust(col_width[1]), row[2].ljust(col_width[2])
            )

    def weekdays_plot(self, consider_holidays: bool = True) -> None:
        dated_demand = self.dated(self.p_el)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if consider_holidays:
            weekdays.append("Holiday")

        fig, axes = plt.subplots(nrows=2, ncols=len(weekdays), figsize=(10, 3), sharey=True)
        axes = zip(*axes)  # transpose
        adder = " / Holidays" if consider_holidays else ""
        fig.suptitle("Weekdays" + adder, fontweight="bold")

        for weekday_number, (ax_tuple, weekday) in enumerate(zip(axes, weekdays)):

            if consider_holidays:
                country = "DE"
                holis = getattr(holidays, country)(years=self.year)
                is_holiday = pd.Series(
                    dated_demand.index.map(lambda x: x in holis), index=dated_demand.index
                )

            if weekday_number <= 6:
                is_given_weekday = dated_demand.index.weekday == weekday_number
                if consider_holidays:
                    ser = dated_demand[is_given_weekday & ~is_holiday]
                else:
                    ser = dated_demand[is_given_weekday]
            else:
                ser = dated_demand[is_holiday]
            df = pd.DataFrame(ser.values.reshape((self.steps_per_day, -1), order="F"))

            df.plot(legend=False, alpha=0.1, color="k", linewidth=1, ax=ax_tuple[0])
            sns.violinplot(y=ser, ax=ax_tuple[1], scale="width", color="lightblue", cut=0)

            ndays = df.shape[1]
            ax_tuple[0].set_title(f"{weekday}\n({ndays})")
            for ax in ax_tuple:
                ax.set_ylabel("$P_{el}$ [kW]")
                ax.get_xaxis().set_visible(False)
                hp.add_thousands_formatter(ax, x=False)

        plt.tight_layout(w_pad=-1.0, h_pad=0)
        sns.despine()

    def averages_plot(self):
        timeframes = ["quarter", "month", "week"]
        fig, axes = plt.subplots(
            ncols=len(timeframes),
            figsize=(10, 1.6),
            gridspec_kw={"width_ratios": [4, 12, 52]},
            sharey=True,
        )
        fig.suptitle("Averages", fontweight="bold")
        plt.tight_layout()

        for timeframe, ax in zip(timeframes, axes):
            ser = self.dated(self.p_el)
            ser = ser.groupby(getattr(ser.index, timeframe)).mean()
            ser.plot.bar(width=0.8, ax=ax, color="darkgray")
            ax.set_ylabel("$P_{el}$ [kW]")
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_title(f"{timeframe.capitalize()}s")
            for i, label in enumerate(ax.xaxis.get_ticklabels()[:-1]):
                if i % 4 != 0:
                    label.set_visible(False)
            hp.add_thousands_formatter(ax, x=False)
        sns.despine()

    def line_plot(self) -> None:
        fig, ax = plt.subplots(1, figsize=(10, 2))
        plt.tight_layout()
        data = self.dated(self.p_el)
        data.plot(linewidth=0.6, ax=ax, color="darkgray")
        ax.set_ylabel("$P_{el}$ [kW]")
        hp.add_thousands_formatter(ax, x=False)
        ax.set_ylim(bottom=0)
        sns.despine()
        ax.set_title("Load curve", fontdict=dict(fontweight="bold"))

    def ordered_line_plot(self) -> None:
        fig, ax = plt.subplots(1, figsize=(10, 2))
        plt.tight_layout()
        data = self.p_el.sort_values(ascending=False).reset_index(drop=True)
        data.plot(linewidth=1.5, ax=ax, color="darkgray")
        ax.set_title("Ordered annual duration curve", fontdict=dict(fontweight="bold"))
        ax.set_ylabel("$P_{el}$ [kW]")
        ax.set_xlabel(f"Time steps [{self.freq_unit}]")
        hp.add_thousands_formatter(ax)
        ax.set_ylim(bottom=0)
        sns.despine()

    def violin_plot(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        plt.tight_layout()
        ax = sns.violinplot(y=self.p_el, cut=0, width=0.4, scale="width", color="lightblue", ax=ax)
        ax.set_ylabel("$P_{el}$ [kW]")
        ax.set_xlim(-0.5, 0.85)
        ax.set_ylim(bottom=0)
        ax.set_title("Metrics", fontdict=dict(fontweight="bold"))
        hp.add_thousands_formatter(ax, x=False)
        ax.get_xaxis().set_visible(False)
        self._annotate_violins_left_side(ax)
        self._annotate_violins_right_side(ax)
        sns.despine(bottom=True)

    def _annotate_violins_left_side(self, ax) -> None:
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

    def _annotate_violins_right_side(self, ax) -> None:
        percentile_range = (50, 60, 70, 80, 90, 95, 97, 99)
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
