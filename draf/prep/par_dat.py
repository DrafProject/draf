from dataclasses import dataclass
from typing import List, Union


@dataclass
class ParDat:
    name: str
    data: Union[str, List]
    doc: str = ""
    src: str = ""
    unit: str = ""

    @property
    def type(self) -> str:
        return self.name.split("_")[0]

    @property
    def component(self) -> str:
        return self.name.split("_")[1]

    @property
    def dims(self) -> str:
        return self.name.split("_")[2]
