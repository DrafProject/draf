[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="doc/images/all.svg" width="200" alt="draf logo">

# **D**emand **r**esponse **a**nalysis **f**ramework

draf is a Python library for analyzing price-based demand response. It is developed by [Markus Fleschutz](https://www.linkedin.com/in/markus-fleschutz/) in a cooperative PhD between [MTU](https://www.mtu.ie/) and the [University of Applied Sciences Karlsruhe](https://www.h-ka.de/en/). An example can be seen in the [Showcase](https://mfleschutz.github.io/draf-showcase/).

# Quick start

```bash
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

``` None
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
- Automatic unit conversion, great descriptions and documentation.
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

You can make your own model...

``` Python
import draf
import gurobipy as gp

cs = draf.CaseStudy(name="foo", year=2019, freq="60min", country="DE")

sc = cs.add_REF_scen()
sc.add_dim("T", infer=True)
sc.prep.add_c_GRID_RTP_T()
sc.prep.add_E_dem_T(profile="G3", annual_energy=5e5)
sc.add_var("C_", unit="€/a", lb=-gp.GRB.INFINITY)

def model_func(d, p, v, m):  # (d)imensions, (p)arameters, (v)ariables, (m)odel
     m.setObjective(v.C_, gp.GRB.MINIMIZE)
     m.addConstr(v.C_ == gp.quicksum(p.E_dem_T[t] * p.c_GRID_RTP_T[t] for t in d.T))

cs.set_model(model_func).optimize().save()
```

... or use an existing one.

``` Python
import draf
import draf.models.PV_BES as mod

cs = draf.CaseStudy(name="ShowCase", year=2017, freq="60min", country="DE")

cs.add_REF_scen(doc="no BES").set_params(mod.params_func).update_params(
     P_PV_CAPx_=100, c_GRID_peak_=50
)

cs.add_scens(
    scen_vars=[("c_GRID_T", "t", ["c_GRID_RTP_T", "c_GRID_TOU_T"]), ("E_BES_CAPx_", "b", [1000])],
    nParetoPoints=4,
)

cs.improve_pareto_and_set_model(mod.model_func).optimize(mod.postprocess_func).save()

```

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


# License

Copyright (c) 2021 Markus Fleschutz

<https://www.gnu.org/licenses/lgpl-3.0.de.html>

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
