from dataclasses import dataclass
from typing import List, Union

from draf import helper as hp


@dataclass
class ParDat:
    name: str
    data: Union[str, List]
    doc: str = ""
    src: str = ""
    unit: str = ""

    @property
    def type(self) -> str:
        return hp.get_type(self.name)

    @property
    def component(self) -> str:
        return hp.get_component(self.name)

    @property
    def acro(self) -> str:
        return hp.get_acro(self.name)

    @property
    def dims(self) -> str:
        return hp.get_dims(self.name)
