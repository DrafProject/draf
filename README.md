[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="doc/images/all.svg" width="200" alt="draf logo">

# **D**emand **r**esponse **a**nalysis **f**ramework

`draf` is a Python library for analyzing price-based demand response.
It is developed by [Markus Fleschutz](https://www.linkedin.com/in/markus-fleschutz/) in a cooperative PhD between [MTU](https://www.mtu.ie/) and the [University of Applied Sciences Karlsruhe](https://www.h-ka.de/en/).
An example can be seen in the [Showcase](https://mfleschutz.github.io/draf-showcase/).

# Quick start

1. Clone the source repository:

   ```sh
   git clone https://github.com/DrafProject/draf
   cd draf
   ```

1. Create and activate conda environment based on environment.yml including editable local version of `draf`:

   ```sh
   conda env create
   conda activate draf
   ```

1. Run tests:

   ```sh
   pytest
   ```

# Features

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
- `draf` can run on Windows, macOS and Linux.

# Usage

For usage examples please see example models in [draf/models](draf/models). Start with [`minimal.py`](draf/models/minimal.py)

# Documentation

## Structure

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

## Data

All data will be placed in the `draf/data`-folder. There are common data that come with the tool.
Local data can be added (e.g. through a symbolic link).

``` None
Raw-data  parquet-files  pd.Series
     |--prep--^   |--get--^
```

## Naming conventions

All parameter and variable names must satisfy the structure `<Type>_<Component>_<Descriptor>_<Dims>`.
E.g. in 'E_GRID_buy_T' `E` is the entity type, `GRID` the component, `buy` the descriptor and `T` the dimension.
Dimensions are denoted with individual capital letters, so `<Dims>` could be `TE` if the entity has the dimensions `T` and `E`.

## Common Abbreviations

| short | long |
|-------|------------------|
| `cs` | CaseStudy object |
| `sc`, `scen` | Scenario object |
| `m`, `mdl` | Model |
| `d`, `dims` | Dimension container object |
| `p`, `params` | Parameters container object |
| `v`, `vars` | Variables container object |
| `r`, `res` | Results container object |
| `ent` | Entity: a variable or parameter |
| `doc` | Documentation / description string |
| `constr` | Constraint |
| `meta` | Meta data |
| `df` | Pandas `DataFrame` |
| `ser` | Pandas `Series` |
| `fp` | file path |
| `gp` | `gurobipy` - the Gurobi Python Interface |
| `XEFs` | Average Electricity Mix Emission Factors |
| `MEFs` | Marginal Power Plant Emission Factors |

# For developers

Bump version using `bump2version` e.g. the patch version can be altered with

```sh
bump2version patch
git push --follow-tags
```

# Status

This piece of software is in a very early stage. Use at your own risk.

# License

Copyright (c) 2021 Markus Fleschutz

<https://www.gnu.org/licenses/lgpl-3.0.de.html>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
