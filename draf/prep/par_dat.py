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
    def etype(self) -> str:
        return hp.get_etype(self.name)

    @property
    def comp(self) -> str:
        return hp.get_component(self.name)

    @property
    def desc(self) -> str:
        return hp.get_desc(self.name)

    @property
    def dims(self) -> str:
        return hp.get_dims(self.name)
