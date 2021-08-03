[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="doc/images/all.svg" width="200" alt="draf logo">

# **D**emand **r**esponse **a**nalysis **f**ramework

draf is a Python library for analyzing price-based demand response. It is developed by [Markus Fleschutz](https://www.linkedin.com/in/markus-fleschutz/) in a cooperative PhD between [MTU](https://www.mtu.ie/) and the [University of Applied Sciences Karlsruhe](https://www.h-ka.de/en/). An example can be seen in the [Showcase](https://mfleschutz.github.io/draf-showcase/).

# Quick start

```sh
git clone https://github.com/DrafProject/draf
cd draf

# creates environment based on environment.yml and installs editable local version:
conda env create

# activate draf environment
conda activate draf  

# Run the tests and ensure that there are no errors
pytest
```

- draf can run on Windows, macOS and Linux.
- Several example models are included with draf under [draf/models](draf/models).
- For the full functionality of draf you need a valid gurobi licence.

# Structure

A `CaseStudy` object can contain several `Scenario` instances e.g.:

```none
CaseStudy
 ⤷ Year, Country, timely resolution (freq)
 ⤷ Plots (scenario comparision, pareto analysis)
 ⤷ Scenario_1
    ⤷ Parameter, Model, Results, Plots
 ⤷ Scenario_2
    ⤷ Parameter, Model, Results, Plots
```

# Some features

- Intuitive handling of complex data structures.
- Fast model formulation with gurobipy.
- Open-source model formulation with pyomo.
- Uses the power of gurobi, the fastest MILP solver available and its community for model formulation and solving.
- Easy and automatic scenario generation and sensitivity analyses.
- Naming conventions for parameters and variables.
- Electricity prices, generation data, load etc. are downloaded on demand and cached for later use.
- Ecological assessment uses dynamic carbon emission factors calculated from historic national electric generation data.
- Economic assessment uses historic day-ahead market prices.
- Modules for load profile creation.
- Modules for peak-load analysis.
- Economic assessment uses historic day-ahead market prices.
- Convenient plotting and presentation functions.
- Automatic unit conversion, descriptions and documentation.
- Uses Python's modern type annotations.
- Whole case studies and individual scenarios can be saved including all results.

## Data

All data will be placed in the `draf/data`-folder. There are common data that come with the tool.
Local data can be added (e.g. through a symbolic link).

``` None
Raw-data  parquet-files  pd.Series
     |--prep--^   |--get--^
```

# Usage

For usage examples please see example models in [draf/models](draf/models). Start with "minimal.py"

# Common Abbreviations

| short | long |
|-------|------------------|
| `cs` | __CaseStudy__ object |
| `sc`, `scen` | __Scenario__ object |
| `m`, `mdl` | __Model__ |
| `d`, `dims` | __Dimension__ container object |
| `p`, `params` | __Parameters__ container object |
| `v`, `vars` | __Variables__ container object |
| `r`, `res` | __Results__ container object |
| `ent` | __entity__: a variable or parameter |
| `doc` | __documentation__ / description string |
| `constr` | __constraint__ |
| `meta` | __meta__ data |
| `df` | pandas __DataFrame__ |
| `ser` | pandas __Series__ |
| `fp` | __file path__ |
| `gp` | __gurobi python__ |
| `XEFs` | Average Electricity __Mix Emission Factors__ |
| `MEFs` | __Marginal__ Power Plant __Emission Factors__ |

# Status

This piece of software is in a very early stage. Use at your own risk.

# Dokumentation

## Conventions

### Naming conventions

All parameters and variable names 

'E_GRID_buy_T'

# License

Copyright (c) 2021 Markus Fleschutz

<https://www.gnu.org/licenses/lgpl-3.0.de.html>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
