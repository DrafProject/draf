from dataclasses import dataclass
from typing import Union

import pandas as pd

from draf import helper as hp


@dataclass
class ParDat:
    name: str
    data: Union[float, pd.Series]
    doc: str = ""
    src: str = ""
    unit: str = ""

    @property
    def type(self) -> str:
        return hp.get_type(self.name)

    @property
    def comp(self) -> str:
        return hp.get_component(self.name)

    @property
    def acro(self) -> str:
        return hp.get_acro(self.name)

    @property
    def dims(self) -> str:
        return hp.get_dims(self.name)
