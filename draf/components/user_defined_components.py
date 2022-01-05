import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from gurobipy import GRB, Model, quicksum

from draf import Collectors, Dimensions, Params, Results, Scenario, Vars
from draf.abstract_component import Component
from draf.conventions import Descs
from draf.helper import conv, get_annuity_factor
from draf.prep import DataBase as db

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class EXAMPLE(Component):
    """An example component"""

    def param_func(self, sc: Scenario):
        pass

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        pass
