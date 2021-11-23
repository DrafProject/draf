"""The draf module provides a toolbox to simulate and optimize energy systems including data
analysis and visualization.
"""

__title__ = "DRAF"
__summary__ = "Demand Response Analysis Framework"
__uri__ = "https://github.com/DrafProject/draf"

__version__ = "0.1.3"

__author__ = "Markus Fleschutz"
__email__ = "mfleschutz@gmail.com"

__license__ = "LGPLv3"
__copyright__ = f"Copyright (C) 2021 {__author__}"

from draf.core.case_study import CaseStudy, open_casestudy, open_latest_casestudy
from draf.core.entity_stores import Dimensions, Params, Results, Vars
from draf.core.scenario import Scenario
from draf.helper import address2coords
