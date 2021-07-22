import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class PeakLoadAnalysis:
    def __init__(self, p_el: np.ndarray):
        self.p_el_np = p_el
        self.steps_per_day = 96
        self.figsize = (25, 3)
        self.freq_unit = "quarter hours"
        self._set_parameters()

    def _set_parameters(self) -> None:
        params = {}
        params["storage_cost_EUR_per_kWh"] = 500
        params["el_price_EUR_per_kWh"] = 0.12
        params["c_el_peak_EUR_per_kW"] = 50.0
        params["P_max_kW"] = self.p_el_np.max()
        params["yearly_sum_kW"] = self.p_el_np.sum() / ((self.steps_per_day / 24))
        params["C_el_EUR_per_a"] = params["el_price_EUR_per_kWh"] * params["yearly_sum_kW"]
        params["C_el_peak_EUR_per_a"] = params["c_el_peak_EUR_per_kW"] * params["P_max_kW"]
        params["C_EUR_per_a"] = params["C_el_EUR_per_a"] + params["C_el_peak_EUR_per_a"]
        self.params = params

    def _get_nPeaks_from_reduction(self, given_reduction: float) -> int:
        for i, j in enumerate(self.p_el_ordered):
            if j < self.p_el_np.max() - given_reduction:
                return i

    def _get_reduc_from_nPeaks(self, nPeaks: int) -> float:
        return self.p_el_np.max() - self.p_el_ordered[nPeaks]

    def histo(
        self, peak_reduction: Optional[float] = None, number_of_peaks: Optional[int] = None
    ) -> plt.Figure:
        """Presents the biggest peaks of a load curve as table and barchart.

        Args:
            peak_reduction: Desired reduction of peak demand in kW.
            number_of_peaks: Desired number of highest peaks to be eliminated.
        """

        self.p_el_ordered = np.sort(self.p_el_np)[::-1]

        if peak_reduction is not None:
            assert peak_reduction > 0
            assert peak_reduction < self.p_el_np.max()
            reduc = peak_reduction
            nPeaks = self._get_nPeaks_from_reduction(peak_reduction)

        elif number_of_peaks is not None:
            assert number_of_peaks > 0
            nPeaks = number_of_peaks
            reduc = self._get_reduc_from_nPeaks(number_of_peaks)

        else:
            raise ValueError(
                f"One of the arguments `peak_reduction` and `number_of_peaks` must be given."
            )

        threshold_power = self.p_el_ordered[nPeaks]
        self.threshold_power = threshold_power
        peak_width_list = np.array([])

        b = 0
        for i in range(len(self.p_el_np)):

            if threshold_power <= self.p_el_np[i] <= self.p_el_np.max():
                b += 1

            else:
                if b > 0:
                    peak_width_list = np.append(peak_width_list, b)
                b = 0

        uni = np.unique(peak_width_list, return_counts=True)

        u = pd.DataFrame(data={"counts": uni[1]}, index=uni[0])
        u.index.name = "peak_width"

        savings = reduc * self.params["c_el_peak_EUR_per_kW"]
        rel_savings = savings / self.params["C_el_peak_EUR_per_a"]
        el_max = self.p_el_np.max()

        string_title = (
            f"{savings:,.2f} €/a ({rel_savings:.1%}) can be saved with a "
            f"{reduc:.0f} kW peak load reduction "
            f"from {el_max:.0f} kW to {threshold_power:.0f} kW."
        )

        string_2 = (
            f"Peak loads above {threshold_power:.0f} kW "
            f"occur in {nPeaks:.0f} quarter hours "
            f"(= {nPeaks / (self.steps_per_day / 24):.0f} hours)."
        )

        n_subplots = 4
        fig, ax = plt.subplots(
            n_subplots, 1, figsize=(self.figsize[0], self.figsize[1] * n_subplots)
        )

        ax[0].plot(self.p_el_np, label="Original load curve", lw=1)
        ax[0].axhline(
            threshold_power, c="r", ls="--", label=f"Threshold = {threshold_power} kW", lw=1
        )
        ax[0].axhline(
            self.p_el_np.max(), c="r", ls="--", label=f"Peak load = {self.p_el_np.max()} kW", lw=1
        )
        ax[0].set(ylabel="Electrical load [kW]", xlabel=f"Time [{self.freq_unit}]")
        ax[0].set_title(string_title, fontsize=14, weight="bold")
        ax[0].legend(loc="lower center", ncol=3)

        ax[1].plot(self.p_el_ordered, label="Ordered load curve", lw=3)
        ax[1].axhline(
            threshold_power, c="r", ls="--", label=f"Threshold = {threshold_power} kW", lw=1
        )
        ax[1].axhline(
            self.p_el_np.max(), c="r", ls="--", label=f"Peak load = {self.p_el_np.max()} kW", lw=1
        )
        ax[1].plot(self.p_el_ordered[:nPeaks], c="r", lw=3)
        ax[1].set(ylabel="Elektrical Load [kW]", xlabel=f"Time [{self.freq_unit}]")
        ax[1].legend(loc="lower center", ncol=3)
        ax[1].annotate(
            string_2,
            xy=(nPeaks, threshold_power),
            xytext=(nPeaks * 0.3, threshold_power * 0.3),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        ax[2].plot(self.p_el_ordered[:nPeaks], marker="o", label=f"{nPeaks} highest peaks", c="r")
        ax[2].set(ylabel="Electrical load [kW]", xlabel=f"Time [{self.freq_unit}]")
        ax[2].legend(loc="lower center")

        label_1 = (
            f"Counts of peak-widths within the {nPeaks:.0f} highest peaks\n"
            f"(e.g. A peak duration of {uni[0][0]:.0f} "
            f"quarter-hours occures {uni[1][0]:.0f} times\n"
            f"whereas a peak duration of {uni[0][-1]:.0f} quarter-hours occurs "
            f"only {uni[1][-1]:.0f} times.)"
        )

        ax[3].bar(uni[0], uni[1], label=label_1, color="r")
        ax[3].set(xlabel=f"Peak duration [{self.freq_unit}]", ylabel="Frequency")
        ax[3].legend(loc="upper center")

        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        return fig

    def simulate(
        self,
        bes_capa: float = 1000.0,
        bes_P_max: float = 1000.0,
        show_res: bool = True,
        threshold: Optional[float] = None,
        transfer_threshold_from_histo: bool = True,
    ) -> None:
        """Simulate an Battery Energy Storage (BES) with a given capacity and maximum power.

        Args:
            bes_capa: Capacity of BES
            bes_P_max: Maximum power of BES
            threshold: Threshold of loading strategy. System tries to load the battery
                 if time series > threshold.
            transfer_threshold_from_histo: If threshold shall be transfered from histo().
        """

        assert bes_capa >= 0
        assert bes_P_max >= 0

        if transfer_threshold_from_histo:
            if hasattr(self, "threshold_power"):
                switch_point = self.threshold_power
            else:
                raise Exception(
                    "If `transfer_threshold_from_histo` is switched on, "
                    "histo() has to be executed first."
                )
        else:
            switch_point = threshold

        p_el = self.p_el_np

        soc = np.zeros(len(p_el))
        p_eex_buy = np.zeros(len(p_el))
        load = np.zeros(len(p_el))
        unload = np.zeros(len(p_el))

        for t, val in enumerate(p_el):
            if t == 0:
                soc[t] = 0
                load[t] = 0
            elif val > switch_point:
                if soc[t - 1] < (bes_capa - (val - switch_point)):
                    load[t] = min(bes_capa - soc[t - 1], val - switch_point, bes_P_max)
                else:
                    load[t] = 0
            elif val < switch_point:
                unload[t] = min(soc[t - 1], switch_point - val, bes_P_max)

            soc[t] = soc[t - 1] + load[t] - unload[t]
            p_eex_buy[t] = val - load[t] + unload[t]

        if show_res:
            self._show_simulation_results(
                p_el, soc, p_eex_buy, load, unload, switch_point, bes_capa, bes_P_max
            )

    def _show_simulation_results(
        self, p_el, soc, p_eex_buy, load, unload, switch_point, bes_capa, bes_P_max
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(p_el, label="P_el", c="r")
        ax.plot(soc, label="SOC")
        ax.plot(p_eex_buy, label="P_eex_buy", c="g")
        ax.plot(load, label="load")
        ax.plot(unload, label="unload")
        ax.plot([switch_point for t in range(len(p_el))], label="switch point")
        ax.plot([p_eex_buy.max() for t in range(len(p_el))], label="EEX_max")

        def get_success():
            if switch_point == p_eex_buy.max():
                return (
                    f"reduce the maximum peak power by "
                    f"{self.p_el_np.max() - p_eex_buy.max():.0f} kW."
                )
            else:
                return (
                    f"only reduce the maximum peak power by "
                    f"{self.p_el_np.max() - p_eex_buy.max():.0f} kW instead of "
                    f"the wanted {self.p_el_np.max() - switch_point:.0f} kW."
                )

        title = (
            f"The battery storage system with {bes_capa:.0f} kWh "
            f"capacity and {bes_P_max: .0f} kW loading/unloading power"
            f"(~{self.params['storage_cost_EUR_per_kWh'] * bes_capa:,.0f} €)"
            f" could {get_success()}"
        )

        ax.set_title(title, y=1.03, fontsize=14, weight="bold")
        fig.legend(ncol=1, loc="center right", bbox_to_anchor=(0.99, 0.5))
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

    def simulate_SA(self) -> None:
        """Sensitivity analysis using the simulate()"""
        for bes_capa in np.arange(3000, 10000, 3000):
            for bes_P_max in np.arange(0, 4000, 1000):
                self.simulate(bes_capa=bes_capa, bes_P_max=bes_P_max, show_res=True)
