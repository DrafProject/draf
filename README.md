<img src="doc/images/all.svg" width="200" alt="draf logo">

---

# **d**emand **r**esponse **a**nalysis **f**ramework (**draf**): a multi-objective decision support and analysis tool for multi-energy hubs with demand response

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/python-3.9-blue?logo=python&logoColor=white)](https://github.com/DrafProject/draf)

`draf` is a ([mixed integer]) [linear programming] optimization framework for local energy systems.

# Quick start

1. Install [miniconda] or [anaconda]

1. Clone the source repository:

   ```sh
   git clone https://github.com/DrafProject/draf
   cd draf
   ```

1. Create and activate a conda environment based on environment.yml including a full editable local version of `draf`:

   ```sh
   conda env create
   conda activate draf
   ```

1. (optional) Run tests:

   ```sh
   pytest
   ```

1. Open Jupyter notebook:

   ```sh
   jupyter notebook
   ```

# Features

![`draf` process](doc/images/draf_process.svg)

- **Time series analysis tools:**
  - `DemandAnalyzer`: analyze energy demand profiles
  - `PeakLoadAnalyzer`: analyze peak loads or run simple battery simulation
- **Pre defined [components](draf/model_builder/components.py):**
  - E.g. Battery energy storage (BES), Battery electric vehicle (BEV), Combined heat and power (CHP), Heat pump (HP), Power to heat (P2H), Photovoltaic (PV), and Thermal energy storage (TES).
  - Sensible naming conventions for parameters and variables, see [Naming conventions](#naming-conventions).
- **Parameter preparation tools:**
  - `TimeSeriesPrepper`: for time series data
    - Electricity prices via [`elmada`]
    - Carbon emission factors via [`elmada`]
    - Standard load profiles from BDEW
    - PV profiles via [`gsee`]
  - [`DataBase`](draf/prep/data_base.py): for scientific data such as cost or efficiency factors.
- **Scenario generation tools:** Easily build individual scenarios or sensitivity analyses.
- **Multi-objective mathematical optimization** with support of different model languages and solvers:
  - [`Pyomo`]: A free and open-source modeling language in Python that supports multiple solvers.
  - [`GurobiPy`]: The Python interface to Gurobi, the fastest MILP solver (see [Mittelmann benchmark]).
- **Plotting tools:** Convenient plots such as heatmaps, tables, pareto plots, etc.
  - support of meta data such as `unit`, `doc`, and `dims`
  - automatic unit conversion
- **Export tools:**
  - `CaseStudy` objects containing all parameters, meta data and results can be saved to files.
  - Data can be exported to [xarray] format.

Other

- Runs on Windows, macOS and Linux.
- Ecological assessment uses dynamic carbon emission factors calculated from historic national electric generation data.
- Economic assessment uses historic day-ahead market prices.

# Usage

For usage examples please see example models in [examples](examples).
Start with [`minimal.py`](examples/gp/minimal.py)

For the usage of Gurobi, a valid Gurobi license is required.

# Documentation

## Structure

A `CaseStudy` object can contain several `Scenario` instances:

![`draf` architecture](doc/images/draf_architecture.svg)

## Naming conventions

All parameter and variable names must satisfy the structure `<Type>_<Component>_<Descriptor>_<Dims>`.
E.g. in 'P_EG_buy_T' `P` is the entity type which stands for electrical power, `EG` the component, `buy` the descriptor and `T` the dimension.
Dimensions are denoted with individual capital letters, so `<Dims>` is `TE` if the entity has the dimensions `T` and `E`.
For examples of types, components, and descriptors please see [conventions.py](draf/conventions.py).

## Common abbreviated programming constructs

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

# For developers

Bump version (replace `<part>` with `major`, `minor`, or `patch`):

```sh
bump2version --dry-run --verbose <part>
bump2version <part>
git push origin <tag_name>
```

Type annotations are used throughout the project.

# Status

This piece of software is in an early stage. Use at your own risk.

# License

Copyright (c) 2021 Markus Fleschutz

`draf` is developed by [Markus Fleschutz] since 2017 in a cooperative PhD between the [Munster Technological University], Ireland and the [Karlsruhe University of Applied Sciences], Germany.

<https://www.gnu.org/licenses/lgpl-3.0.de.html>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<!-- SOURCES -->
[`elmada`]: https://github.com/DrafProject/elmada
[`gsee`]: https://github.com/renewables-ninja/gsee
[`GurobiPy`]: https://pypi.org/project/gurobipy
[`Pyomo`]: https://github.com/Pyomo/pyomo
[anaconda]: https://www.anaconda.com/products/individual
[linear programming]: https://en.wikipedia.org/wiki/Linear_programming
[Markus Fleschutz]: https://linktr.ee/m.fl
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[Mittelmann benchmark]: http://plato.asu.edu/ftp/milp.html
[mixed integer]: https://en.wikipedia.org/wiki/Integer_programming
[Munster Technological University]: https://www.mtu.ie
[Karlsruhe University of Applied Sciences]: https://www.h-ka.de/en
[xarray]: http://xarray.pydata.org/en/stable
