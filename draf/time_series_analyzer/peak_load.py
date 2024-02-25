import logging
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from draf import helper as hp
from draf.core.datetime_handler import DateTimeHandler
from draf.core.entity_stores import Params

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class PeakLoadAnalyzer(DateTimeHandler):
    def __init__(self, p_el: pd.Series, year: int, freq: str, figsize=(10, 3)):
        self.p_el = p_el
        self._set_dtindex(year=year, freq=freq)
        self.figsize = figsize
        self.params = Params()
        self.set_prices()

    def set_prices(self, c_EG=0.12, c_EG_peak=50.0):
        p = self.params
        p.c_EG = c_EG
        p.c_EG_peak = c_EG_peak
        p.p_max = self.p_el.max()
        p.e_annual_sum = self.p_el.sum() * self.step_width
        p.C_EG = p.c_EG * p.e_annual_sum
        p.C_EG_peak = p.c_EG_peak * p.p_max
        p.C = p.C_EG + p.C_EG_peak

    def histo(self, target_percentile: int = 95):
        """Presents the biggest peaks of a load curve as table and barchart.

        Args:
            peak_reduction: Desired reduction of peak demand in kW.
            number_of_peaks: Desired number of highest peaks to be eliminated.
        """
        assert 1 < target_percentile < 100, "target_percentile must be in [1..100]"
        p_el = self.p_el
        self.target_percentile = target_percentile
        self.p_el_ordered = p_el_ordered = p_el.sort_values(ascending=False).reset_index(drop=True)
        self.target_peakload = target_peakload = p_el.quantile(target_percentile / 100)
        self.trimmed_peaks = trimmed_peaks = p_el_ordered[p_el_ordered > target_peakload]
        self.nPeaks = len(trimmed_peaks)

        fig, ax = plt.subplots(nrows=4, figsize=(self.figsize[0], self.figsize[1] * 5))
        fig.suptitle(self._get_stats(self.target_percentile), fontweight="bold")
        fig.set_facecolor("whitesmoke")
        self.load_curve_with_threshold_plot(ax[0])
        self.ordered_peaks_plot(ax[1])
        self.zoomed_ordered_peaks_plot(ax[2])
        self.peak_width_bar_plot(ax[3])
        sns.despine()
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.6)

    def _get_stats(self, target_percentile):
        reduc = self.p_el_ordered[0] - self.target_peakload
        savings_raw = reduc * self.params.c_EG_peak
        rel_savings = savings_raw / self.params.C_EG_peak
        savings, savings_unit = hp.auto_fmt(savings_raw, "€/a")
        return textwrap.dedent(
            f"""\
            Peak load reduction of {reduc:,.0f} kW
            from {self.p_el.max():,.0f} to {self.target_peakload:,.0f} kW ({target_percentile:.0f} percentile)
            
            Savings:{rel_savings:.1%} network costs (={savings:,.2f} {savings_unit} with {self.params.c_EG_peak:,.0f} €/kW)
        """
        )

    def load_curve_with_threshold_plot(self, ax):
        ax.plot(self.p_el, label="Original load curve", lw=0.6, c="darkgray")
        ax.axhline(
            self.target_peakload,
            c="firebrick",
            ls="--",
            label=f"Threshold = {self.target_peakload:,.0f} kW",
            lw=1,
        )
        ax.set(ylabel="$P_{el}$ (kW)")
        ax.set_ylim(bottom=0, top=self.p_el.max())
        ax.set_title("Load curve", fontweight="bold")
        ax.margins(y=0.0, x=0.0)
        ax.legend(loc="lower center", ncol=3)
        hp.add_thousands_formatter(ax)

    def ordered_peaks_plot(self, ax):
        trimmed_peaks = self.trimmed_peaks
        target_peakload = self.target_peakload
        p_el_ordered = self.p_el_ordered
        nPeaks = self.nPeaks
        ax.add_patch(
            patches.Rectangle(
                (0, target_peakload),
                trimmed_peaks.size,
                self.p_el.max() - target_peakload,
                linewidth=0.5,
                color="firebrick",
                alpha=0.1,
            )
        )
        ax.plot(p_el_ordered, lw=2, c="darkgray")
        ax.set_title("Ordered load duration curve", fontweight="bold")
        ax.plot(trimmed_peaks, c="firebrick", lw=2)
        ax.margins(y=0.0, x=0.0)
        ax.set(ylabel="$P_{el}$ (kW)", xlabel=f"Time ({self.freq_unit})")
        ax.annotate(
            textwrap.dedent(
                f"""\
                Peak loads above {target_peakload:,.0f} kW
                occur in {nPeaks:,.0f} time steps (≈ {nPeaks / (self.steps_per_day / 24):,.0f} hours)
                """
            ),
            xy=(nPeaks, target_peakload),
            xytext=(len(p_el_ordered) * 0.02, p_el_ordered[0] * 0.01),
            arrowprops=dict(
                facecolor="lightgray", shrink=0.05, linewidth=0, width=2, headwidth=8, headlength=8
            ),
        )
        ax.set_ylim(bottom=0, top=p_el_ordered[0])
        ax.margins(y=0.0)
        hp.add_thousands_formatter(ax)

    def zoomed_ordered_peaks_plot(self, ax):
        ax.patch.set_facecolor("firebrick")
        ax.patch.set_alpha(0.1)
        plt.tight_layout()
        self.trimmed_peaks.plot(markersize=5, linewidth=3, color="firebrick", ax=ax)
        ax.set_title(f"Zoom on {self.nPeaks:,.0f} highest peaks", fontweight="bold")
        ax.set(ylabel="$P_{el}$ (kW)", xlabel=f"Time ({self.freq_unit})")
        hp.add_thousands_formatter(ax, x=False)

    def peak_width_bar_plot(self, ax):
        ser = self._get_peak_widths()
        ser.plot.bar(
            width=0.9,
            label=textwrap.dedent(
                f"""\
                Counts of peak-widths within the {self.nPeaks:,.0f} highest peaks
                (e.g. A peak duration of {ser.index[0]:,.0f} time steps occures {ser.iloc[0]:,.0f} times)"""
            ),
            color="firebrick",
            ax=ax,
        )
        ax.set(xlabel=f"Peak duration ({self.freq_unit})", ylabel="Frequency")
        ax.set_title("Peak durations", fontweight="bold")
        ax.legend(loc="upper right")
        hp.add_thousands_formatter(ax, x=False)

    def _get_peak_widths(self):
        p_el_np = self.p_el.values
        target_peakload = self.target_peakload
        peak_width_list = np.array([])
        b = 0
        for i in range(len(p_el_np)):
            if target_peakload < p_el_np[i] <= p_el_np.max():
                b += 1
            else:
                if b > 0:
                    peak_width_list = np.append(peak_width_list, b)
                b = 0
        uni = np.unique(peak_width_list, return_counts=True)
        return pd.Series(data=uni[1], index=uni[0].astype(int))

    def simulate_BES(
        self,
        e_bes_capa: float = 1000.0,
        p_bes_max: float = 1000.0,
        c_bes_inv: float = 500.0,
        threshold: Optional[float] = None,
        transfer_threshold_from_histo: bool = True,
    ) -> None:
        """Simulate a Battery Energy Storage (BES) with a given capacity and maximum power.

        Args:
            e_bes_capa: Capacity of BES in kWh
            p_bes_max: Maximum power of BES in kW
            c_bes_inv: Investment costs of BES in € / kWh_el
            threshold: Threshold of charging strategy. System tries to charge the battery
                if time series > threshold.
            transfer_threshold_from_histo: If threshold shall be transfered from histo().
        """

        assert e_bes_capa >= 0
        assert p_bes_max >= 0

        if transfer_threshold_from_histo:
            if hasattr(self, "target_peakload"):
                switch_point = self.target_peakload
            else:
                raise Exception(
                    "If `transfer_threshold_from_histo` is switched on, "
                    "histo() has to be executed first."
                )
        else:
            switch_point = threshold

        p_el = self.p_el.values

        E_BES_T = np.zeros(len(p_el))
        p_buy = np.zeros(len(p_el))
        load = np.zeros(len(p_el))
        unload = np.zeros(len(p_el))

        for t, val in enumerate(p_el):
            if t == 0:
                E_BES_T[t] = e_bes_capa
                load[t] = 0
            elif val > switch_point:
                if E_BES_T[t - 1] < (e_bes_capa - (val - switch_point)):
                    load[t] = min(e_bes_capa - E_BES_T[t - 1], val - switch_point, p_bes_max)
                else:
                    load[t] = 0
            elif val < switch_point:
                unload[t] = min(E_BES_T[t - 1], switch_point - val, p_bes_max)

            E_BES_T[t] = E_BES_T[t - 1] + (load[t] - unload[t]) * (24 / self.steps_per_day)
            p_buy[t] = val - load[t] + unload[t]

        # Plot results
        fig, ax = plt.subplots(3, figsize=(10, 7), sharex=True, sharey=True)
        ax[0].plot(p_el, label="P_buy (old)", c="tab:red", lw=1)
        ax[0].plot(p_buy, label="P_buy (new)", c="tab:green", lw=1)
        ax[0].plot([switch_point for t in range(len(p_el))], label="switch point", lw=1, c="grey")
        ax[0].plot([p_buy.max() for t in range(len(p_el))], label="new_max", lw=1, c="k")
        ax[1].plot(load, label="load", lw=1)
        ax[1].plot(-unload, label="unload", lw=1)
        ax[2].plot(E_BES_T, label="E_BES_T", lw=1)

        def get_success():
            if switch_point == p_buy.max():
                return f"reduce the maximum peak power by {self.p_el.max() - p_buy.max():,.0f} kW."
            else:
                return (
                    "only reduce the maximum peak power by "
                    f"{self.p_el.max() - p_buy.max():,.0f} kW instead of "
                    f"the wanted {self.p_el.max() - switch_point:,.0f} kW."
                )

        title = (
            f"The BESS ({e_bes_capa:,.0f} kWh, {p_bes_max:,.0f} kW,"
            f" ~{c_bes_inv * e_bes_capa:,.0f} €) could\n{get_success()}"
        )

        for x in ax:
            x.margins(0)

        ax[0].set_ylabel("Power (kW)")
        ax[1].set_ylabel("Power (kW)")
        ax[2].set_ylabel("Energy (kWh)")

        fig.suptitle(title, y=1.03, fontsize=12, weight="bold")
        fig.legend(ncol=1, loc="center right", bbox_to_anchor=(1.15, 0.5))
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
        sns.despine()

    def simulate_SA(self) -> None:
        """Sensitivity analysis using the simulate()"""
        for e_bes_capa in np.arange(3000, 10000, 3000):
            for p_bes_max in np.arange(0, 4000, 1000):
                self.simulate_BES(e_bes_capa=e_bes_capa, p_bes_max=p_bes_max)
