import logging
from abc import ABC
from typing import List, Optional, Tuple, Union

import pandas as pd

from draf import helper as hp

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class DateTimeHandler(ABC):
    @property
    def step_width(self) -> float:
        """Returns the step width of the current datetimeindex.
        e.g. 0.25 for a frequency of 15min."""
        return hp.get_step_width(self.freq)

    def dt(self, k: int, g: int) -> float:
        if hasattr(self, "segmentation"):
            if self.segmentation:
                # TSA was conducted WITH segmentation
                return self.hoursPerSegment[k, g]
            else:
                # TSA was conducted WITHOUT segmentation
                return self.step_width
        else:
            # TSA was NOT conducted
            return self.step_width

    @property
    def dt_info(self) -> str:
        """Get an info string of the chosen time horizon of the case study."""
        t1_str = f"{self.dtindex_custom[0].day_name()}, {self.dtindex_custom[0]}"
        t2_str = f"{self.dtindex_custom[-1].day_name()}, {self.dtindex_custom[-1]}"
        return (
            f"t1 = {self._t1:<5} ({t1_str}),\n"
            f"t2 = {self._t2:<5} ({t2_str})\n"
            f"Length = {self.dtindex_custom.size}"
        )

    @property
    def steps_per_day(self):
        steps_per_hour = 60 / hp.int_from_freq(self.freq)
        return int(steps_per_hour * 24)

    @property
    def n_days(self):
        return int(len(self.dtindex) / self.steps_per_day)

    @property
    def freq_unit(self):
        if self.freq == "15min":
            return "1/4 h"
        elif self.freq == "30min":
            return "1/2 h"
        elif self.freq == "60min":
            return "h"

    def match_dtindex(
        self, data: Union[pd.DataFrame, pd.Series], resample: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        if resample:
            data = self.resample(data)
        return data[self._t1 : self._t2 + 1]

    def resample(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return hp.resample(
            data, year=self.year, start_freq=hp.estimate_freq(data), target_freq=self.freq
        )

    def _set_dtindex(self, year: int, freq: str) -> None:
        assert year in range(1980, 2100)
        self.year = year
        self.freq = freq
        self.dtindex = hp.make_datetimeindex(year=year, freq=freq)
        self.dtindex_custom = self.dtindex
        self._t1 = 0
        self._t2 = self.dtindex.size - 1  # =8759 for a normal year

    def _get_int_loc_from_dtstring(self, s: str) -> int:
        return self.dtindex.get_loc(f"{self.year}-{s}")

    def _get_first_int_loc_from_dtstring(self, s: str) -> int:
        x = self._get_int_loc_from_dtstring(s)
        try:
            return x.start
        except AttributeError:
            return x

    def _get_last_int_loc_from_dtstring(self, s: str) -> int:
        x = self._get_int_loc_from_dtstring(s)
        try:
            return x.stop
        except AttributeError:
            return x

    def _get_integer_locations(self, start, steps, end) -> Tuple[int, int]:
        t1 = self._get_first_int_loc_from_dtstring(start) if isinstance(start, str) else start
        if steps is not None and end is None:
            assert t1 + steps < self.dtindex.size, "Too many steps are given."
            t2 = t1 + steps - 1
        elif steps is None and end is not None:
            t2 = self._get_last_int_loc_from_dtstring(end) if isinstance(end, str) else end
        elif steps is None and end is None:
            t2 = self.dtindex.size - 1
        else:
            raise ValueError("One of steps or end must be given.")
        return t1, t2

    def timeslice(self, start: Optional[str], stop: Optional[str]) -> "Slice":
        """Get timeslice from start and stop strings.

        Example slicing from 17th to 26th of August
            >>> ts = cs.timeslice("8-17", "8-26")
            >>> sc.params.c_EG_T[ts].plot()
        """
        start_int = None if start is None else self._get_first_int_loc_from_dtstring(start)
        stop_int = None if stop is None else self._get_last_int_loc_from_dtstring(stop)
        return slice(start_int, stop_int)

    def dated(
        self, data: Union[pd.Series, pd.DataFrame], activated=True
    ) -> Union[pd.Series, pd.DataFrame]:
        """Add datetime index to a data entity.

        The frequency and year are taken from the CaseStudy or the Scenario object.

        Args:
            data: A pandas data entity.
            activated: If False, the data is returned without modification.

        """
        if activated:
            assert isinstance(
                data, (pd.Series, pd.DataFrame)
            ), f"No data given, but type {type(data)}"
            data = data.copy()
            dtindex_to_use = self.dtindex[data.index.min() : data.index.max() + 1]
            data.index = dtindex_to_use
        return data
