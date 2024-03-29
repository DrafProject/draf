# DRAF CHEAT SHEET

## Common work flow

```python
# import draf and its pre-defined component templates:
import draf
from draf.components import *

# create a case study:
cs = draf.CaseStudy("my_case_study", year=2019, country="DE", freq="60min",
                    coords=(49.01, 8.39), consider_invest=True)

# define a modeling horizon (default: whole year, here: 2 days):
cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)

# create and parametrize a reference scenario:
sc = cs.add_REF_scen("REF", components=[BES, Main, EG, eDem, PV])
sc.update_params(E_BES_CAPx_=100)

# create a second scenario based on "REF" and update a parameter:
sc_new = cs.add_scen("new_scen", based_on="REF")
sc_new.update_params(E_BES_CAPx_=500)

# solve all scenarios:
cs.optimize()

# save the case study (including all scenarios and data) to your hard disk:
cs.save()
```

```python
cs = draf.open_latest_casestudy("my_case_study")

# get an overview of the results:
cs.plot()
```

## Interactive analysis

- `cs.scens` - show scenarios
- `sc.dims` - show dimensions
- `sc.params` - show parameter entities
- `sc.vars` - show variable entities
- `sc.res` - show result entities

### On a case study

- `cs.scens.REF` - access a specific scenario
- `cs.plot()` - access essential plots and tables
- `cs.plot.describe_interact()` - description of all entities
- `cs.optimize(solver_params=dict(MIPGap=0))` - set a [MIPGap](https://www.gurobi.com/documentation/9.5/refman/mipgap2.html)

### On a scenario

- `sc.params.c_EG_RTP_T` - access a specific parameter entity
- `sc.res.P_EG_buy_T` - access a specific results entity
- `sc.plot.describe()` - description of all entities of this scenario

### On an entity

- `cs.dated(my_entity_T)` - add a date-time index to a time series
- `my_entity_T.plot()` - plot an entity with [pd.Series.plot](https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html)

### Paths

- `draf.paths.DATA_DIR` - draf data directory
- `draf.paths.RESULTS_DIR` - draf results directory
- `draf.paths.CACHE_DIR` - draf cache directory
- `elmada.paths.CACHE_DIR` - elmada cache directory

### [Elmada](https://github.com/DrafProject/elmada)

- `import elmada`
- `elmada.get_prices(method="hist_EP")` - get historic day-ahead market prices from ENTSO-E
- `elmada.get_emissions(method="XEF_EP")` - get historic grid-mix emission factors

## Building own components

```python
from draf import Collectors, Dimensions, Params, Results, Scenario, Vars
from gurobipy import GRB, Model, quicksum
from draf.prep import DataBase as db

# short version
class MyShortComponent:
    def param_func(self, sc): ...
    def model_func(self, sc, m, d, p, v, c): ...

# advanced version 
class MyAdvancedComponent(Component):
    """Description of your own component"""

    def dim_func(self, sc: Scenario):
        sc.dim("F", ["ng", "bio"], doc="Types of fuel")

    def param_func(self, sc: Scenario):
        sc.collector("F_fuel_F", doc="Fuel power", unit="kWh")
        sc.param(from_db=db.c_Fuel_F)
        sc.var("C_Fuel_ceTax_", doc="Total carbon tax on fuel", unit="k€/a")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstr(v.C_Fuel_ == p.k__dT_ * v.F_fuel_F.prod(p.c_Fuel_F) * conv("€", "k€", 1e-3))
        c.CE_TOT_["Fuel"] = v.CE_Fuel_
        c.P_EL_source_T["PV"] = lambda t: v.P_PV_FI_T[t] + v.P_PV_OC_T[t]
```

### param_func

- `sc.collector("P_EL_source_T", doc="...", unit="kW_el")` - add a collector
- `sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)` - define a variable that can be negative
- `sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")` - define a variable with time dimension
- `sc.param("c_PV_inv_", data=200, doc="Investment costs", unit="€/kW_p", src="my_data_source")`
- `sc.param(from_db=db.c_Fuel_F)` - use data from the [draf database](draf/prep/data_base.py)
- prepper functions (default parameter name is the function name):
  - `sc.prep.c_EG_T()` - real-time-prices-tariffs
  - `sc.prep.ce_EG_T()` - dynamic carbon emission factors
  - `sc.prep.P_eDem_T(profile="G1", annual_energy=5e6)` - electricity demand with 5 GWh/a
  - `sc.prep.dQ_hDem_T(annual_energy=2e6, target_temp=22.0, threshold_temp=15.0)` - heating demand with 2 GWh/a using weather data
  - `sc.prep.P_PV_profile_T()` - photovoltaic profile
  - `sc.prep.c_EG_addon_(AbLa_surcharge=0.00003, Concession_fee=0.0011, ...)` - electricity price components other than wholesale prices
  - `sc.prep.T__amb_T()` - ambient air temperature from nearest weather station

### model_func

- General
  - `d.T` - time index (special dimension)
  - `p.k__dT_` - time step (special parameter)
  - `p.k__PartYearComp_` - weighting factor to compensate part year analysis (special parameter)
- [GurobiPy Syntax](https://www.gurobi.com/documentation), e.g.:
  - `from gurobipy import GRB, Model, quicksum` - import GurobiPy objects
  - `m.setObjective((...), GRB.MINIMIZE)` - set objective
  - `m.addConstr((v.your == p.constraint * v.goes + p.here), "constr_1")` - add one constraint
  - `m.addConstrs((v.your_T[t] == p.constraint_T[t] * v.goes_ + p.here_T[t] for t in d.T), "constr_2")` - add a set of constraints
- [Pyomo Syntax](http://www.pyomo.org/documentation), e.g.:
  - `import pyomo.environ as pyo` - import Pyomo
  - `pyo.Objective(expr=(...), sense=pyo.minimize)` - set objective
  - `m.constr_1 = pyo.Constraint(expr=(v.your == p.constraint * v.goes + p.here))` - add one constraint
  - `m.constr_1 = pyo.Constraint(d.T, rule=lambda t: v.your_T[t] == p.constraint_T[t] * v.goes_ + p.here_T[t])` - add a set of constraints

## Helper functions

- `draf.helper.address2coords("Moltkestraße 30, Karlsruhe")` - converts an address to geo coordinates
- `draf.helper.play_beep_sound()` - play a beep sound (is used in the `cs.optimize()` routine)
- `draf.helper.read(...)` - draf default read function
- `draf.helper.write(...)` - draf default write function
- `pd.read_csv(...)` - read an external csv file

## Naming conventions

### Naming conventions for entities

In general, entity names should adhere to the following structure:
`<Etype>_<Component>_<OptionalDescription>_<DIMENSIONS>`

|Examples:|entity 1|entity 2|
|-|-|-|
|Entity name|`P_EG_buy_T`|`c_PV_inv_`|
|Etype|`P`|`c`|
|Component|`EG`|`PV`|
|Description|`buy`|`inv`|
|Dimension|`T`|`-`|

For typical symbols, see [draf/conventions.py](draf/conventions.py).

### Naming conventions for Python objects

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

## Versioning of draf

Bump version (replace `<part>` with `major`, `minor`, or `patch`):

```sh
bump2version --dry-run --verbose <part>
bump2version <part>
git push origin <tag_name>
```
