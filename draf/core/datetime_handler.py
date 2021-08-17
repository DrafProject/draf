import logging
from abc import ABC
from typing import List, Tuple, Union

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
    def freq_unit(self):
        if self.freq == "15min":
            return "1/4 h"
        elif self.freq == "30min":
            return "1/2 h"
        elif self.freq == "60min":
            return "h"

    def trim_to_datetimeindex(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        return data[self._t1 : self._t2 + 1]

    def _set_dtindex(self, year: int, freq: str) -> None:
        assert year in range(1980, 2100)
        self.year = year
        self.freq = freq
        self.dtindex = hp.make_datetimeindex(year=year, freq=freq)
        self.dtindex_custom = self.dtindex
        self._t1 = 0
        self._t2 = self.dtindex.size - 1  # =8759 for a normal year

    def _get_datetime_int_loc_from_string(self, s: str) -> int:
        return self.dtindex.get_loc(f"{self.year}-{s}")

    def _get_integer_locations(self, start, steps, end) -> Tuple[int, int]:
        t1 = self._get_datetime_int_loc_from_string(start) if isinstance(start, str) else start
        if steps is not None and end is None:
            assert t1 + steps < self.dtindex.size, "Too many steps are given."
            t2 = t1 + steps - 1
        elif steps is None and end is not None:
            t2 = self._get_datetime_int_loc_from_string(end) if isinstance(end, str) else end
        elif steps is None and end is None:
            t2 = self.dtindex.size - 1
        else:
            raise ValueError("One of steps or end must be given.")
        return t1, t2

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
            data = data.copy()
            dtindex_to_use = self.dtindex[data.index.min() : data.index.max() + 1]
            data.index = dtindex_to_use
        return data
