<img src="doc/images/all.svg" width="200" alt="draf logo">

<br>

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![python](https://img.shields.io/badge/python-3.9-blue?logo=python&logoColor=white)](https://github.com/DrafProject/draf)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**d**emand **r**esponse **a**nalysis **f**ramework (`draf`) is an analysis and decision support framework for local multi-energy hubs focusing on demand response.
It uses the power of ([mixed integer]) [linear programming] optimization, [`pandas`], [`plotly`], [`matplotlib`], [`elmada`], [`gsee`] and more to help users along the energy system analysis process.

## Features

![`draf` process](doc/images/draf_process.svg)

- **Time series analysis tools:**
  - `DemandAnalyzer` - analyze energy demand profiles
  - `PeakLoadAnalyzer` - analyze peak loads or run simple battery simulation
- **Easily parameterizable [component templates](draf/components/component_templates.py):**
  - E.g. Battery energy storage (BES), Battery electric vehicle (BEV), Combined heat and power (CHP), Heat pump (HP), Power-to-heat (P2H), Photovoltaic (PV), and Thermal energy storage (TES).
  - Sensible naming conventions for parameters and variables, see section [Naming conventions](#naming-conventions).
- **Parameter preparation tools:**
  - `TimeSeriesPrepper` - for time series data
    - Electricity prices via [`elmada`]
    - Carbon emission factors via [`elmada`]
    - Standard load profiles from [BDEW]
    - PV profiles via [`gsee`] (In Germany, using weather data from [DWD])
  - [`DataBase`](draf/prep/data_base.py) - for scientific data such as cost or efficiency factors.
- **Scenario generation tools:** Easily build individual scenarios or sensitivity analyses.
- **Multi-objective mathematical optimization** with support of different model languages and solvers:
  - [`Pyomo`] - A free and open-source modeling language in Python that supports multiple solvers.
  - [`GurobiPy`] - The Python interface to Gurobi, the fastest MILP solver (see [Mittelmann benchmark]).
- **Plotting tools:** Convenient plots such as heatmaps, tables, pareto plots, etc. using [`plotly`], [`matplotlib`] and [`seaborn`].
  - Support of meta data such as `unit`, `doc`, `src`, and `dims`
  - Automatic unit conversion
- **Export tools:**
  - `CaseStudy` objects containing all parameters, meta data and results can be saved to files.
  - Data can be exported to [xarray] format.

## Quick start

`draf` runs on Windows, macOS and Linux.
For the usage of Gurobi, a valid Gurobi license is required.

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

1. Have a look at the [examples](examples).
  Start with the [`minimal`](examples/minimal.py) example.

## For users

### Structure

A `CaseStudy` object can contain several `Scenario` instances:

![`draf` architecture](doc/images/draf_architecture.svg)

### Naming conventions

All parameter and variable names must satisfy the structure `<Type>_<Component>_<Descriptor>_<Dims>`.
E.g. in 'P_EG_buy_T', `P` is the entity type which stands for electrical power, `EG` the component, `buy` the descriptor and `T` the dimension.
Dimensions are denoted with individual capital letters, so `<Dims>` is `TE` if the entity has the dimensions `T` and `E`.
See [conventions.py](draf/conventions.py) for examples of types, components, and descriptors.

## Contributing

Contributions in any form are welcome! Please contact [Markus Fleschutz] and have a look [here](for_devs.md).

## License and status

Copyright (c) 2022 Markus Fleschutz

License: [LGPL v3]

The development of `draf` was initiated by [Markus Fleschutz] in 2017 and continued in a cooperative PhD between the [MeSSO Research Group] of the [Munster Technological University], Ireland and the [Energy System Analysis Research Group] of the [Karlsruhe University of Applied Sciences], Germany.
Thank you [Markus Bohlayer], [Adrian Bürger], and [Andre Leippi] for your valuable feedback.
<img src="doc/images/MTU_HKA_Logo.svg" width="500" alt="MTU_HKA_Logo">

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<!-- SOURCES -->
[`elmada`]: https://github.com/DrafProject/elmada
[`gsee`]: https://github.com/renewables-ninja/gsee
[`GurobiPy`]: https://pypi.org/project/gurobipy
[`matplotlib`]: https://matplotlib.org
[`pandas`]: https://pandas.pydata.org
[`plotly`]: https://plotly.com
[`Pyomo`]: https://github.com/Pyomo/pyomo
[`seaborn`]: https://seaborn.pydata.org
[Adrian Bürger]: https://scholar.google.de/citations?user=UcLkLlEAAAAJ
[anaconda]: https://www.anaconda.com/products/individual
[Andre Leippi]: https://www.linkedin.com/in/andre-leippi-3187a81a7
[BDEW]: https://www.bdew.de
[DWD]: https://www.dwd.de
[Energy System Analysis Research Group]: https://www.h-ka.de/en/ikku/energy-system-analysis
[Karlsruhe University of Applied Sciences]: https://www.h-ka.de/en
[LGPL v3]: https://www.gnu.org/licenses/lgpl-3.0.de.html
[linear programming]: https://en.wikipedia.org/wiki/Linear_programming
[Markus Bohlayer]: https://scholar.google.com/citations?user=hH1FQVsAAAAJ
[Markus Fleschutz]: https://mfleschutz.github.io
[MeSSO Research Group]: https://messo.cit.ie
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[Mittelmann benchmark]: http://plato.asu.edu/ftp/milp.html
[mixed integer]: https://en.wikipedia.org/wiki/Integer_programming
[Munster Technological University]: https://www.mtu.ie
[xarray]: http://xarray.pydata.org/en/stable
