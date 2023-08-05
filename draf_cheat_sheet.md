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

# optional time series aggregation
cs.aggregate_temporally(
    n_typical_periods=10,
    n_steps_per_period=24,
    segmentation=True,
    n_segments_per_period=5,
    cluster_method="hierarchical",
    store_TSA_instance=True,
    weight_dict=None,
)

# solve all scenarios:
cs.optimize()

# After optimization, save the case study (including all scenarios and data) to your hard disk:
cs.save()
```

```python
cs = draf.open_latest_casestudy("my_case_study")

# get an overview of the results:
cs.plot()
```

Note:

* Most case study functions can be chained, e.g., `cs.aggregate_temporally().optimize().save()`.
* We recommend saving only after the optimization run, since some dynamic objects for the optimization cannot be pickled.

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

- `sc.params.c_EG_RTP_KG` - access a specific parameter entity
- `sc.res.P_EG_buy_KG` - access a specific results entity
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

class MyComponent(Component):
    """Describe your component here."""

    order = 5  # determines when this component is called compared to others

    def dim_func(self, sc: Scenario):
        sc.dim("F", ["ng", "bio"], doc="Types of fuel")

    def param_func(self, sc: Scenario):
        sc.collector("F_fuel_F", doc="Fuel power", unit="kWh")
        sc.param(from_db=db.c_Fuel_F)
        sc.var("C_Fuel_ceTax_", doc="Total carbon tax on fuel", unit="k€/a")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstr(v.C_Fuel_ == quicksum(v.F_fuel_F[f] *p.c_Fuel_F[f] for f in d.F) * conv("€", "k€", 1e-3))
        c.CE_TOT_["Fuel"] = v.CE_Fuel_
        c.P_EL_source_KG["PV"] = lambda t: v.P_PV_FI_KG[k, g] + v.P_PV_OC_KG[k, g]

    def postprocess_func(self, sc: Scenario):
        """This function is called after the optimization run."""
        sc.res.make_pos_ent("P_EG_buy_KG")
```

### `dims_func`

- `sc.dim("A", data=[1, 2, 3] doc="Consumer")` - add a dimension. Access via `sc.dims.A`.

### `param_func`

- `sc.collector("P_EL_source_T", doc="...", unit="kW_el")` - add a collector. Access via `sc.collectors.P_EL_source_KG`.
- `sc.var("C_TOT_", ...)` - add a variable. Access via `sc.vars.C_TOT_`.
- `sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)` - define a variable that can be negative
- `sc.var("P_PV_FI_T", doc="Feed-in", unit="kW_el")` - define a variable with time dimension. The sets for the variables are inferred by the dimension part of the variable name, e.g., the variable `"P_PV_FI_T"` has `len(sc.dims.T)` data points but the variable `"P_PV_FI_TY"` would have  `len(sc.dims.T) * len(sc.dims.Y)` data points.
- `sc.param("c_PV_inv_", data=200, doc="Investment costs", unit="€/kW_p", src="my_data_source")`
- `sc.param(from_db=db.c_Fuel_F)` - use data from the [draf database](draf/prep/data_base.py)
- prepper functions (default parameter name is the function name):
  - `sc.prep.c_EG_KG()` - real-time-prices-tariffs. Access via `sc.params.c_EG_KG`.
  - `sc.prep.ce_EG_KG()` - dynamic carbon emission factors
  - `sc.prep.P_eDem_KG(profile="G1", annual_energy=5e6)` - electricity demand with 5 GWh/a
  - `sc.prep.dQ_hDem_KG(annual_energy=2e6, target_temp=22.0, threshold_temp=15.0)` - heating demand with 2 GWh/a using weather data
  - `sc.prep.P_PV_profile_KG()` - photovoltaic profile
  - `sc.prep.c_EG_addon_(AbLa_surcharge=0.00003, Concession_fee=0.0011, ...)` - electricity price components other than wholesale prices
  - `sc.prep.T__amb_KG()` - ambient air temperature from nearest weather station based on the given geo coordinates

### `model_func`

- General
  - `d.T` - time index
  - `p.k__dT_` - static time step width
  - `p.k__PartYearComp_` - weighting factor to compensate part year analysis
- [GurobiPy Syntax](https://www.gurobi.com/documentation), e.g.:
  - `from gurobipy import GRB, Model, quicksum` - import GurobiPy objects
  - `m.setObjective((...), GRB.MINIMIZE)` - set objective
  - `m.addConstr((v.your == p.constraint * v.goes + p.here), "constr_1")` - add one constraint
  - `m.addConstrs((v.your_KG[k, g] == p.constraint_KG[k, g] * v.goes_ + p.here_KG[k, g] for k in d.K for g in d.G), "constr_2")` - add a set of constraints
- [Pyomo Syntax](http://www.pyomo.org/documentation), e.g.:
  - `import pyomo.environ as pyo` - import Pyomo
  - `pyo.Objective(expr=(...), sense=pyo.minimize)` - set objective
  - `m.constr_1 = pyo.Constraint(expr=(v.your == p.constraint * v.goes + p.here))` - add one constraint
  - `m.constr_1 = pyo.Constraint(d.K, d.G, rule=lambda t: v.your_KG[k, g] == p.constraint_KG[k, g] * v.goes_ + p.here_KG[k, g])` - add a set of constraints

### `postprocess_func`

- `sc.res.make_pos_ent("P_EG_buy_KG")` - trim slightly negative result values that sometime occure due to numerical inaccuracies and may cause issues in plots the expect positive values.

### After optimization

After optimization, the values of the variable `sc.vars.P_PV_FI_T` can be accessed through `sc.res.P_PV_FI_T`.

### Time series aggregation (TSA)

The [dev-TSA](https://github.com/DrafProject/draf/tree/dev-TSA) branch supports time series aggregation.
For this, `sc.aggregate_temporally()` must be called before optimization.
A TSA-supporting model is based on typical periods (`k in d.K`) and intra-period time steps or segments (`g in d.G`) instead of the conventional time steps (`t in d.T`).
Also dynamic time step widths (`sc.dt(k, g)`) are used instead of the static time steps (`p.k__dT_`).
Even seasonal storages [can be modeled](https://arxiv.org/abs/1710.07593) with a few modifications.

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

In the dimensions, the `T` (time steps), respectively, the `KG` (typical days and segments) come first.
Then the other dimensions according to the ascending alphabet

|Examples:|entity 1|entity 2| entity 3|
|-|-|-|-|
|Entity name|`P_EG_buy_T`|`c_PV_inv_`|`c_TEST_test_KGABC`|
|Etype|`P`|`c`|`c`|
|Component|`EG`|`PV`|`TEST`|
|Description|`buy`|`inv`|`test`|
|Dimension|`T`|`-`|`KGABC`|

For typical symbols, see [draf/conventions.py](draf/conventions.py).

### Photovoltaic data

For Germany only, the `PV-prep` module can generate generate a specific PV energy generation profile (kW/kWp) using [GSEE](https://github.com/renewables-ninja/gsee) with data from the nearest with [DWD](https://www.dwd.de) weather stations that have available data for ambient air temperature and solar irradiation for that year:

```python
coords = draf.helper.address2coords("Moltkestraße 30, Karlsruhe")
config = dict(coords=coords, year=2021)
draf.prep.pv.get_pv_power(**config)

# Output:
2021-01-01 00:00:00    0.0
2021-01-01 01:00:00    0.0
                      ... 
2021-12-31 22:00:00    0.0
2021-12-31 23:00:00    0.0
Length: 8760, dtype: float64
```

Shows more info on the chosen DWD weather stations:

```python
draf.prep.pv.get_nearest_stations(**config)

# Output:
#                             solar    air_temperature
# Stations_id                  5906               4177
# von_datum                19790101           20081101
# bis_datum                20230531           20230625
# Stationshoehe                  98                116
# geoBreite                 49.5063            48.9726
# geoLaenge                  8.5584             8.3301
# Stationsname             Mannheim       Rheinstetten
# Bundesland      Baden-Württemberg  Baden-Württemberg
# distance_in_km          55.813722           6.586742
```

Shortcut: `sc.prep.P_PV_profile_KG()` generates and adds a PV profile to your scenario `sc` using `coords` and `year` from your case study.

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
