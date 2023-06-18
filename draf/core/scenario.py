from __future__ import annotations

import copy
import datetime
import inspect
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gurobipy as gp
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from tsam.timeseriesaggregation import TimeSeriesAggregation

from draf import helper as hp
from draf import paths
from draf.abstract_component import Component
from draf.conventions import Etypes
from draf.core.datetime_handler import DateTimeHandler
from draf.core.draf_base_class import DrafBaseClass
from draf.core.entity_stores import Collectors, Dimensions, Params, Results, Vars
from draf.core.mappings import GRB_OPT_STATUS, VAR_PAR
from draf.core.time_series_prepper import TimeSeriesPrepper
from draf.plotting import ScenPlotter
from draf.prep.data_base import ParDat
from draf.time_series_analyzer.demand_analyzer import DemandAnalyzer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class Scenario(DrafBaseClass, DateTimeHandler):
    """An energy system configuration scenario.

    Args:
        id: A short unique string for the scenario. This is used to access the scenario and to label
            scenarios in plots with many scenarios. Don't start with a number to be able to access
            it through `cs.scens.my_id`.
        freq: Default time step. E.g. '60min'
        year: Year
        country: Country code of the case study location. Used for the preparation of parameters
            such as dynamic electricity prices.
        name: A longer, more telling string for the scenario used for plotting.
        doc: A detailed description of the scenario used for plotting and documentation.
        dtindex: Pandas DateTime index for the preparation of parameters.
        dtindex_custom: Pandas DateTime index that defines the optimization time frame.
        t1, t2: Start and end of optimization time frame as hour of a year.
        coords: Geographic coordinates (lattitude, longitude) of the case study location used for
            preparing weather-based parameters data such as photovoltaico profiles, air
            temperatures, heat demands, etc.
        cs_name: The name of respective case study.
        components: List of component objects. This can be either a component class or an instance of it.
        custom_model: A `model_func` that is called within the optimization routine.
        consider_invest: If investments should be considered.
        mdl_language: The modeling language used. Either 'gp' or 'pyo'.
        obj_vars: The names of the obje ctive variables.
        update_dims: A dictionary with dimension-data pairs to update existing dimensions.
    """

    def __init__(
        self,
        id: str = "",
        year: int = 2019,
        freq: str = "60min",
        country: str = "DE",
        name: str = "",
        doc: str = "",
        dtindex: Optional[pd.DatetimeIndex] = None,
        dtindex_custom: Optional[pd.DatetimeIndex] = None,
        t1: Optional[int] = None,
        t2: Optional[int] = None,
        coords: Optional[Tuple[float, float]] = None,
        cs_name: str = "no_case_study",
        components: Optional[List[Union[Component, type]]] = None,
        custom_model: Optional[Callable] = None,
        consider_invest: bool = False,
        mdl_language="gp",
        obj_vars=("C_TOT_", "CE_TOT_"),
        update_dims: Optional[Dict] = None,
    ):
        self.id = id
        self.name = name
        self.doc = doc
        self.country = country
        self.consider_invest = consider_invest
        self.coords = coords
        self.cs_name = cs_name
        self.mdl_language = mdl_language
        self.custom_model = custom_model

        self.dims = Dimensions()
        self.params = Params()
        self.plot = ScenPlotter(sc=self)
        self.prep = TimeSeriesPrepper(sc=self)
        self.vars = Vars()
        self.collectors = Collectors()
        self.obj_vars = obj_vars

        if dtindex is None and dtindex_custom is None and t1 is None and t2 is None:
            self._set_dtindex(year=year, freq=freq)
        else:
            self.year = year
            self.dtindex = dtindex
            self.dtindex_custom = dtindex_custom
            self._t1 = t1
            self._t2 = t2
            self.freq = freq

        self.dim("T", data=list(range(self._t1, self._t2 + 1)), doc=f"Time steps")

        # The default is to have just one period (containing all time steps) and one typical period.
        # TSA-objects are used for consistency.
        self.dim("I", data=list(range(1)), doc="Candidate period index")
        self.dim("K", data=list(range(1)), doc="Typical period index")
        self.periodsOrder = [0]  # mapping candidate periods to typical periods
        self.periodOccurrences = [1]
        self.dim("G", data=self.dims.T, doc="Intra-period step index")
        self.param(
            "n__stepsPerPeriod_",
            data=len(self.dims.T),
            doc="Number of time steps per typical period",
        )

        self.prep.k__PartYearComp_()
        self.prep.k__dT_()

        if components is not None:
            for i, comp in enumerate(components):
                if inspect.isclass(comp):
                    components[i] = comp()
                assert isinstance(
                    components[i], Component
                ), f"The component at position {i} is invalid."
            if len(components) > 1:
                components = sorted(components, key=lambda k: k.order)
            self.add_components(components, update_dims)

    def __getstate__(self) -> Dict:
        """Remove objects with dependencies for serialization with pickle."""
        state = self.__dict__.copy()
        state.pop("mdl", None)
        state.pop("plot", None)
        state.pop("prep", None)
        state.pop("custom_model", None)
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self.plot = ScenPlotter(sc=self)
        self.prep = TimeSeriesPrepper(sc=self)

    def __repr__(self):
        return self._make_repr(
            excluded=[
                "dims",
                "params",
                "vars",
                "dtindex",
                "dtindex_custom",
                "collectors",
                "components",
                "res",
            ]
        )

    @property
    def size(self) -> str:
        """The file size of the scenario."""
        return hp.human_readable_size(hp.get_size(self))

    @property
    def _has_feasible_solution(self) -> bool:
        """If the scenario has a feasible solution."""
        return hasattr(self, "res")

    @property
    def _is_optimal(self) -> bool:
        """If the results were optimal."""
        if self.mdl_language == "gp":
            return self.mdl.Status == gp.GRB.OPTIMAL
        elif self.mdl_language == "pyo":
            return self._termination_condition == pyo.TerminationCondition.optimal
        else:
            RuntimeError("`mdl_language` must be 'gp' or 'pyo'.")

    def info(self) -> None:
        print(self._make_repr())

    def _make_repr(self, excluded: Optional[Iterable] = None):
        """Get overview of attributes of the scenario object."""
        if excluded is None:
            excluded = []
        preface = "<{} object>".format(self.__class__.__name__)
        attribute_list = []
        for k, v in self.get_all().items():
            if k in excluded:
                v = "[...]"
            attribute_list.append(f"• {k}: {v}")
        return "{}\n{}".format(preface, "\n".join(attribute_list))

    def _set_time_trace(self) -> None:
        self._time = time.perf_counter()

    def _get_time_diff(self) -> float:
        return time.perf_counter() - self._time

    @property
    def _res_fp(self) -> Path:
        """Returns the path to the case study's default result directory."""
        fp = paths.RESULTS_DIR / self.cs_name
        fp.mkdir(exist_ok=True)
        return fp

    @property
    def _all_ents_dict(self) -> Dict:
        """Returns a name:data Dict of all entities."""
        d = self.params.get_all()
        if self._has_feasible_solution:
            d.update(self.res.get_all())
        return d

    def filter_entities(
        self,
        etype: Optional[str] = None,
        comp: Optional[str] = None,
        desc: Optional[str] = None,
        dims: Optional[str] = None,
        func: Optional[Callable] = None,
        params: bool = True,
        vars: bool = True,
    ) -> Dict[str, Union[float, pd.Series]]:
        """Return a dictionary with filtered entities.

        Args:
            etype: The component part of the entity name.
            comp: The component part of the entity name, e.g. `BES`.
            desc: The description part ot the entity name, e.g. `in`.
            dims: The dimension part of the entity name, e.g. `T`.
            func: Function that takes the entity name and returns a if the entity
                should be included or not.
            params: If parameters are included.
            vars: If variable results are included.
        """
        kwargs = dict(etype=etype, comp=comp, desc=desc, dims=dims, func=func)
        d = dict()
        if params:
            d.update(self.params.filtered(**kwargs))
        if self._has_feasible_solution and vars:
            d.update(self.res.filtered(**kwargs))
        return d

    def get_mdpv(self) -> Tuple[gp.Model, Dimensions, Params, Vars]:
        return (self.mdl, self.dims, self.params, self.vars)

    def _set_default_solver_params(self) -> None:
        defaults = {
            "LogToConsole": 0,
            "OutputFlag": 1,
            "LogFile": str(self._res_fp / "gurobi.log"),
            "MIPGap": 0.1,
            "MIPFocus": 1,
        }

        for param, value in defaults.items():
            self.mdl.setParam(param, value)

    def update_par_dic(self) -> None:
        self._par_dic = self.params._to_dims_dic()

    def get_total_energy(self, data: pd.Series) -> float:
        """Get the total energy over the optimization time frame of a power entity with
        the index `T`.
        """
        try:
            return data.sum() * self.step_width
        except AttributeError as e:
            return np.nan

    gte = get_total_energy

    @property
    def par_dic(self) -> Dict:
        """Creates the par_dic at the first use then caches it. Use `update_par_dic()` to update."""
        if not hasattr(self, "_par_dic") or self.params._changed_since_last_dic_export:
            self._par_dic = self.params._to_dims_dic()
        return self._par_dic

    def update_res_dic(self):
        self._res_dic = self.res._to_dims_dic()

    def analyze_demand(self, time_series: pd.Series) -> DemandAnalyzer:
        da = DemandAnalyzer(p_el=time_series, year=self.year, freq=self.freq)
        da.show_stats()
        return da

    @property
    def res_dic(self):
        """Creates the res_dic at the first use then caches it. Use `update_res_dic()` to update."""
        if not hasattr(self, "_res_dic") or self.res._changed_since_last_dic_export:
            try:
                self._res_dic = self.res._to_dims_dic()
                return self._res_dic
            except AttributeError:
                logger.warning("No res_dic available.")
                return None
        return self._res_dic

    def get_var_par_dic(self, what: str) -> Dict:
        """Returns the dictionary for parameters or for results."""
        what_long = VAR_PAR[what]
        var_par_dic = getattr(self, what_long, dict())
        if var_par_dic is None:
            raise RuntimeError(f"Scenario has no {what_long}.")
        else:
            return var_par_dic

    def _get_entity_store(self, what: str) -> object:
        """Return the entity store of an entity type."""
        if what == "p":
            return self.params
        elif what == "v":
            return self.res
        else:
            raise RuntimeError(f"`what` must be either 'p' or 'v' not {what}.")

    def yield_all_ents(self):
        """A generator that yields Name, Data tuples of all entities."""
        for k, v in self._all_ents_dict.items():
            yield k, v

    def get_ent(self, ent: str) -> Union[float, pd.Series]:
        """Get entity-data by its name and NAN if entity is not available."""
        try:
            return self.get_entity(ent)
        except (KeyError, AttributeError):
            return np.nan

    def get_entity(self, ent: str) -> Union[float, pd.Series]:
        """Get entity-data by its name."""
        try:
            return getattr(self.res, ent)
        except AttributeError:
            return getattr(self.params, ent)

    def _get_entity_type(self, ent: str) -> str:
        """Return 'p' if 'ent' is a parameter and 'v' if ent is a variable."""
        if ent in self.params.get_all():
            return "p"
        elif ent in self.res.get_all():
            return "v"
        else:
            raise AttributeError(f"Entity {ent} not found.")

    def set_params(self, params_builder_func: Callable) -> Scenario:
        """Executes the params builder function to fill the params object and the variables
        meta-informations.
        """
        self._set_time_trace()

        try:
            params_builder_func(sc=self)

        except RuntimeError as e:
            logger.error(e)

        self._update_time_param("t__params_", "Time to build params", self._get_time_diff())
        return self

    def add_components(self, components: List, update_dims: Optional[Dict] = None):
        """Add components to the scenario."""
        self._set_time_trace()
        logger.info(f"Set params for scenario {self.id}")

        for comp in components:
            comp.dim_func(sc=self)

        if update_dims is not None:
            for k, v in update_dims.items():
                self.dim(name=k, data=v, update=True)

        for comp in components:
            logger.debug(f" ⤷ component {comp.__class__.__name__}")
            comp.param_func(sc=self)
        self._update_time_param("t__params_", "Time to build params", self._get_time_diff())

        self.components = components

    def set_model(
        self,
        custom_model_func: Optional[Callable] = None,
        custom_model_func_loc: int = 0,
        speed_up: bool = True,
    ) -> Scenario:
        """Instantiates an optimization model on top of the given parameters and
         meta-informations for variables. Models of components are automatically fetched.

        Args:
            custom_model_func: A `model_func` that is additionally executed.
            custom_model_func_loc: Order of the custom_model_func. Determines when given
                custom_model_func is executed compared to the other components' `model_func`s.
            speed_up: If speed increases should be exploited by converting the parameter objects to
                tuple-dicts before building the constraints.
        """
        # TODO: The "factory" design pattern may be suitable to cover _instantiate_model and
        #       activate_vars (https://refactoring.guru/design-patterns/factory-method)
        self._instantiate_model()

        self._set_time_trace()
        self._activate_vars()
        self._update_time_param("t__vars_", "Time to activate variables", self._get_time_diff())

        self._set_time_trace()

        logger.info(f"Set model for scenario {self.id}.")

        # Put all model functions in the correct order
        model_func_list = []
        if hasattr(self, "components"):
            model_func_list += [comp.model_func for comp in self.components]
        if custom_model_func is not None:
            model_func_list.insert(custom_model_func_loc, custom_model_func)
        if hasattr(self, "custom_model"):
            if self.custom_model is not None:
                model_func_list.append(self.custom_model)

        # Execute all model functions
        for model_func in model_func_list:
            self.execute_model_func(model_func, speed_up=speed_up)

        self._update_time_param("t__model_", "Time to build model", self._get_time_diff())
        return self

    def execute_model_func(self, model_func, speed_up=True):
        """Sets a model function to a scenario. The model needs to be set before."""
        p = (
            self.get_tuple_dict_container(self.params)
            if speed_up and self.mdl_language == "gp"
            else self.params
        )
        model_func(sc=self, m=self.mdl, d=self.dims, p=p, v=self.vars, c=self.collectors)

    def _update_time_param(self, ent_name: str, doc: str, time_in_seconds: float):
        try:
            ent = getattr(self.params, ent_name)
            self.param(name=ent_name, data=ent + time_in_seconds, update=True)
        except AttributeError:
            self.param(name=ent_name, data=time_in_seconds, doc=doc, unit="seconds")

    def get_tuple_dict_container(self, params) -> Params:
        """Returns a copy of the params object where all Pandas Series objects are converted
         to gurobipy's tupledicts in order to speed up the execution of the model_func.

        Meta data are not copied.
        """
        td = Params()
        for name, obj in params.get_all().items():
            if isinstance(obj, pd.Series):
                data = gp.tupledict(obj.to_dict())
            else:
                data = obj
            setattr(td, name, data)
        return td

    def _instantiate_model(self) -> None:
        if self.mdl_language == "gp":
            self.mdl = gp.Model(name=self.id)
            self._set_default_solver_params()

        elif self.mdl_language == "pyo":
            self.mdl = pyo.ConcreteModel(name=self.id)

    def _activate_vars(self) -> Scenario:
        """Instantiate variables according to the meta-data in `vars._meta`."""
        if self.mdl_language == "gp":
            self._activate_gurobipy_vars()

        elif self.mdl_language == "pyo":
            self._activate_pyomo_vars()

        return self

    def _activate_gurobipy_vars(self) -> None:
        for name, metas in self.vars._meta.items():
            if "vtype" not in metas:
                continue  # this prevents activating variables created by `res.make_pos_ent()``

            kwargs = dict(lb=metas["lb"], ub=metas["ub"], name=name, vtype=metas["vtype"])

            if hasattr(self, "mdl"):
                mdl = self.mdl
            else:
                raise RuntimeError(
                    "The scenario has no model yet. Please first instantiate the model"
                    " e.g. with `sc.set_model` or `sc._instantiate_model`."
                )

            if metas["is_scalar"]:
                var_obj = mdl.addVar(**kwargs)

            else:
                dims = hp.get_dims(name)
                dims_list = self.get_coords(dims=dims)
                var_obj = mdl.addVars(*dims_list, **kwargs)

            setattr(self.vars, name, var_obj)

    def _activate_pyomo_vars(self) -> None:
        def get_domain(vtype: str):
            vtype_mapper = {"C": pyo.Reals, "B": pyo.Binary, "I": pyo.Integers}
            return vtype_mapper[vtype]

        for name, metas in self.vars._meta.items():

            kwargs = dict(
                bounds=(metas.get("lb"), metas.get("ub")),
                within=get_domain(metas.get("vtype")),
                initialize=metas.get("initialize"),
            )

            if metas["is_scalar"]:
                var_obj = pyo.Var(**kwargs)

            else:
                dims = hp.get_dims(name)
                dims_list = self.get_coords(dims=dims)
                var_obj = pyo.Var(*dims_list, **kwargs)

            setattr(self.vars, name, var_obj)
            setattr(self.mdl, name, var_obj)

    def optimize(
        self,
        logToConsole: bool = False,
        outputFlag: bool = True,
        show_results: bool = False,
        keep_vars: bool = True,
        postprocess_func: Optional[Callable] = None,
        which_solver="gurobi",
        solver_params: Optional[Dict] = None,
        scale_KG_to_T: bool = True,
    ) -> Scenario:
        """Solves the optimization problem and does postprocessing if the function is given.
        Results are stored in the Results-object of the scenario.

        Args:
            logToConsole: Sets the LogToConsole param to the gurobipy model.
            outputFlag: Sets the outputFlag param to the gurobipy model.
            show_results: If the Cost and Carbon emissions are printed.
            keep_vars: If the variable objects are kept after optimization run.
            postprocess_func: Function which is executed with the scenario as argument.
            which_solver: Choose solver. Only applicable when using Pyomo.
            solver_params: Set solver params such as MIPGap, MIPFocus or LogFile.
        """
        for k, v in self.params.get_all().items():
            warn_if_data_contains_nan(data=v, name=k)

        if not hasattr(self, "mdl"):
            self.set_model()

        kwargs = dict(logToConsole=logToConsole, outputFlag=outputFlag, show_results=show_results)

        logger.info(f"Optimize")
        if self.mdl_language == "gp":
            assert which_solver == "gurobi"
            solution_exists = self._optimize_gurobipy(**kwargs, solver_params=solver_params)

        elif self.mdl_language == "pyo":
            solution_exists = self._optimize_pyomo(**kwargs, which_solver=which_solver)

        if hasattr(self, "collectors") and solution_exists:
            self._cache_collector_values()
        if not keep_vars:
            self.vars.delete_all()

        if scale_KG_to_T:
            logger.info(f"Scale KG data to T.")
            self._scale_all_KG_data_to_T(self.params, self.params)
            self._scale_all_KG_data_to_T(self.res, self.res)

        pp_funcs = []
        if postprocess_func is not None:
            pp_funcs.append(postprocess_func)
        if hasattr(self, "components"):
            for comp in self.components:
                if hasattr(comp, "postprocess_func"):
                    pp_funcs.append(comp.postprocess_func)

        for postprocess_func in pp_funcs:
            postprocess_func(self)

        return self

    def _scale_all_KG_data_to_T(self, source, destination) -> None:
        """`source` and `destination` can either be `params` or `res`."""
        for k, v in source.filtered(func=lambda name: "KG" in hp.get_dims(name)).items():
            if self.has_TSA:
                data = self._predict_T_after_TSA(v)
            else:
                data = hp.from_simple_KG_to_T_format(v)
            new_name = hp.rename_KG_to_T(k)
            destination._meta[new_name] = source._meta[k]
            if my_dims := destination._get_meta(new_name, "dims"):
                destination._set_meta(new_name, "dims", my_dims.replace("KG", "T"))
            setattr(destination, new_name, data)

    def _predict_T_after_TSA(self, ser: pd.Series, rename: bool = True) -> pd.Series:
        ndims = ser.index.nlevels
        if ndims > 2:
            df = ser.unstack(list(range(2, ndims)))
            return df.apply(self._predict_T_after_TSA, rename=False).stack(list(range(ndims - 2)))
        else:
            name = ser.name
            data_KG = ser.unstack().values
            ser = pd.Series(
                [
                    data_KG[k, g]
                    for k in self.periodsOrder
                    for g, n_steps in self.timeStepsPerSegment[k].items()
                    for _ in range(n_steps)
                ]
                if self.segmentation
                else [data_KG[k, g] for k in self.periodsOrder for g in self.dims.G],
                name=name,
            )
            ser = ser.rename_axis(index=["T"] + ser.index.names[1:])
            if rename:
                ser = hp.rename_data_from_KG_to_T(ser)
            return ser

    def _optimize_gurobipy(self, logToConsole, outputFlag, show_results, solver_params) -> bool:
        logger.info(f"Optimize {self.id}")
        self._set_time_trace()

        solver_params_dic = {"LogToConsole": int(logToConsole), "OutputFlag": int(outputFlag)}
        if solver_params is not None:
            solver_params_dic.update(solver_params)
        for k, v in solver_params_dic.items():
            self.mdl.setParam(k, v)

        self.mdl.optimize()
        self._update_time_param("t__solve_", "Time to solve the model", self._get_time_diff())
        status = self.mdl.Status
        status_str = GRB_OPT_STATUS[status]
        if (status == gp.GRB.OPTIMAL) or (status == gp.GRB.TIME_LIMIT and self.mdl.SolCount > 0):
            self.res = Results(self)
            if status == gp.GRB.TIME_LIMIT:
                logger.warning("Time-limit reached")
            if show_results:
                try:
                    results = ", ".join([f"{x}={self.res.get(x):.2f}" for x in self.obj_vars])
                    print(f"{self.id}: {results}({status_str})")
                except ValueError:
                    s = " or ".join(self.obj_vars)
                    logger.warning(f"{s} not found.")
        elif status in [gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE]:
            response = input(
                "The model is infeasible. Do you want to compute an Irreducible"
                " Inconsistent Subsystem (IIS)?\n[(y)es / (n)o] + ENTER"
            )
            if response == "y":
                self.calculate_IIS()
        elif status == gp.GRB.TIME_LIMIT and self.mdl.SolCount == 0:
            logger.warning("Time limit reached withouth feasible solution.")
        else:
            raise RuntimeError(
                f"ERROR solving scenario {self.name}: mdl.Status="
                f" {status} ({GRB_OPT_STATUS[status]}) --> {self.mdl.Params.LogFile}"
            )
        solution_exists = self.mdl.SolCount > 0
        return solution_exists

    def _optimize_pyomo(self, logToConsole, outputFlag, show_results, which_solver) -> bool:
        logger.info(f"Optimize {self.id}")
        self._set_time_trace()
        solver = pyo.SolverFactory(which_solver)
        logfile = str(self._res_fp / "pyomo.log") if outputFlag else None
        results = solver.solve(self.mdl, tee=logToConsole, logfile=logfile)
        self._update_time_param("t__solve_", "Time to solve the model", self._get_time_diff())
        status = results.solver.status
        tc = results.solver.termination_condition
        self._termination_condition = tc
        if status == pyo.SolverStatus.ok:
            self.res = Results(self)
            if tc == pyo.TerminationCondition.maxTimeLimit:
                logger.warning("Time-limit reached")
            if show_results:
                try:
                    print(
                        f"{self.id}: C_TOT_={self.res.C_TOT_:.2f},"
                        f" CE_TOT_={self.res.CE_TOT_:.2f} ({tc})"
                    )
                except ValueError:
                    logger.warning("res.C_TOT_ or res.CE_TOT_ not found.")
        else:
            raise RuntimeError(
                f"ERROR solving scenario {self.name}: status= {status}, ",
                f"termination condition={tc}) --> logfile: {logfile}",
            )
        solution_exists = status == pyo.SolverStatus.ok
        return solution_exists

    def calculate_IIS(self) -> None:
        """Calculate and print the Irreducible Inconsistent Subsystem (IIS)."""
        self.mdl.computeIIS()
        if self.mdl.IISMinimal:
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
        print("\nThe following constraint(s) cannot be satisfied:")
        for c in self.mdl.getConstrs():
            if c.IISConstr:
                print("%s" % c.constrName)

    def _special_copy(self) -> Scenario:
        """Returns a deepcopy of this scenario that preserves the pointer to the case study."""
        scen_copy = copy.deepcopy(self)
        scen_copy.plot.sc = scen_copy
        scen_copy.prep.sc = scen_copy
        return scen_copy

    def _get_now_string(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def export_model(self, filetype="lp", fp=None):
        """Exports the optimization problem to a file e.g. an lp-file. If no filepath (fp) is
        given, the file is saved in the case study's default result directory.
        """
        date_time = self._get_now_string()
        if fp is None:
            fp = self._res_fp / f"{date_time}_{self.id}.{filetype}"

        self.mdl.write(str(fp))
        logger.info(f"written to {fp}")

    def save_results(self) -> None:
        """Saves the scenario results to a pickle-file."""
        fp = self._res_fp / f"{self.id}.p"
        with open(fp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved scenario results to {fp}")

    def save(self, name: Optional[str] = None) -> None:
        """Saves the scenario to a pickle-file."""
        date_time = self._get_now_string()

        name = f"{date_time}_{self.id}.p" if name is None else f"{name}.p"

        fp = self._res_fp / name

        try:
            with open(fp, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved scenario to {fp}")

        except pickle.PicklingError as e:
            logger.error(
                f"PicklingError {e}: Try deactivate Ipython's autoreload to save the scenario."
            )

    def get_xarray_dataset(self, include_vars: bool = True, include_params: bool = True):
        """Get an xarray dataset with all parameters and results."""
        import xarray as xr

        loop_list = []

        if include_vars and (self.res_dic is not None):
            loop_list.append(self.res_dic)

        if include_params and (self.par_dic is not None):
            loop_list.append(self.par_dic)

        ds = xr.auto_combine(
            [
                value.to_xarray()
                for scen_dic in loop_list
                for dim, value in scen_dic.items()
                if len(dim) > 0
            ]
        )
        return ds

    def get_coords(self, dims: str) -> List:
        """Returns a list of coordinates for the given dimensions."""
        return [getattr(self.dims, dim) for dim in list(dims)]

    def var(
        self,
        name: str,
        doc: str = "",
        unit: str = "",
        lb: float = 0.0,
        ub: float = 1e100,
        vtype: str = "C",
    ) -> None:
        """Add metadata of one or more variables to the scenario.

        Note:
            * This does not yet create a gurobipy or pyomo-variable-object.
            * Dims are inferred from name suffix
        """
        dims = hp.get_dims(name)
        is_scalar = dims == ""
        self._warn_if_unexpected_unit(name, unit)
        self.vars._meta[name] = dict(
            doc=doc,
            unit=unit,
            vtype=vtype,
            lb=lb,
            ub=ub,
            etype=hp.get_etype(name),
            comp=hp.get_component(name),
            dims=dims,
            is_scalar=is_scalar,
        )

    def update_var_bound(self, name: str, lb: Optional[float] = None, ub: Optional[float] = None):
        """Update upper or lower bound of an optimization variable."""
        if lb is not None:
            self.vars._meta[name].update(lb=lb)
        if ub is not None:
            self.vars._meta[name].update(ub=ub)
        return self

    def update_upper_bound(self, name: str, upper_bound: float):
        self.vars._meta[name].update(ub=upper_bound)
        return self

    def update_lower_bound(self, name: str, lower_bound: float):
        self.vars._meta[name].update(lb=lower_bound)
        return self

    def _get_idx(self, ent_name: str) -> Union[List, pd.MultiIndex]:
        dims = hp.get_dims(ent_name)
        coords = self.get_coords(dims)
        if len(dims) == 1:
            idx = coords[0]
        else:
            idx = pd.MultiIndex.from_product(coords, names=list(dims))
        return idx

    def get_unit(self, ent_name: str) -> Optional[str]:
        """Get the unit metadata for a given entity."""
        return self.get_meta(ent_name=ent_name, meta_type="unit")

    def get_doc(self, ent_name: str) -> Optional[str]:
        """Get the doc metadata for a given entity."""
        return self.get_meta(ent_name=ent_name, meta_type="doc")

    def get_src(self, ent_name: str) -> Optional[str]:
        """Get the src metadata for a given entity."""
        return self.get_meta(ent_name=ent_name, meta_type="src")

    def get_meta(self, ent_name: str, meta_type: str) -> Optional[str]:
        """Get meta-information such as doc or unit for a given entity.

        Note:
            Meta-information are stored as followed:
            sc.res (obj)
            sc.res._meta (dict)
            sc.res._meta[<entity-name>] (dict with metas {"doc":..., "unit":...})
        """
        for attr in ["params", "res", "dims", "collectors"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                metas = obj._meta.get(ent_name, "")
                if metas != "":
                    return metas.get(meta_type, "")
        return None

    def update_params(self, **kwargs) -> Scenario:
        """Update multiple existing parameters.

        - Converts scalar value to uniform series if set to a multi-dimensional entity.
        - Converts `.._T`-time series to the `.._KG` format.

        Example usage: `sc.update_params(c_EG_peak_=0, P_EG_dem_KG=2000, c_EG_KG=c_EG_RTP_T)`
        """
        for ent_name, data in kwargs.items():

            if isinstance(data, str):
                try:
                    data = self.params.get(data)
                except AttributeError as e:
                    raise e

            if not hasattr(self.params, ent_name):
                raise RuntimeError(f"The parameter {ent_name} you want to update does not exist.")

            if hp.fits_convention(ent_name, data):
                self.param(ent_name, data=data, update=True)
            else:
                self.param(ent_name, fill=data, update=True)

        return self

    def fix_vars(self, **kwargs) -> Scenario:
        """Fix optimization variables.
        Example: `sc.fix_vars(P_PV_CAPn_=100, P_WT_CAPn_=2000)`
        """
        for ent_name, data in kwargs.items():
            self.update_var_bound(name=ent_name, lb=data, ub=data)
        return self

    def param(
        self,
        name: Optional[str] = None,
        data: Optional[Union[int, float, list, Dict, np.ndarray, pd.Series]] = None,
        doc: str = "",
        unit: str = "",
        src: str = "",
        fill: Optional[float] = None,
        update: bool = False,
        from_db: Optional[ParDat] = None,
        set_param: bool = True,
    ) -> pd.Series:
        """Add or update a parameter of the scenario.

        Args:
            name: The entity name. It has to end with an underscore followed by the
                single-character dimensions i.e. a solely time-dependent parameter has
                to end with `_T` e.g. `P_eDem_T`;
                a scalar has to end with `_` e.g. `C_TOT_inv_`.
            data: Data is normally given as int, float or pd.Series. Lists, Dicts and
                np.ndarrays are accepted and converted to pd.Series. Lists and np.ndarrays
                must have the same length as dims.T.
            doc: A description string for the parameter.
            fill: If a float is given here, the series is filled for all relevant dimensions
                inferred from the name.
            update: If True, the meta-data will not be touched, just the data changed.
            from_db: DataBase object.

        """
        if from_db is not None:
            d = from_db.__dict__.copy()
            if name is not None:
                d.update(name=name)
            if doc != "":
                d.update(doc=doc)
            return self.param(**d)

        assert name is not None

        dims = hp.get_dims(name)
        comp = hp.get_component(name)
        etype = hp.get_etype(name)

        if dims == "":
            assert isinstance(data, (float, int)), (
                f"'{name}' has trailing underscore in the name. "
                f"So it indicates a scalar entity but is a {type(data)}-type."
            )
        else:
            assert (
                dims == dims.upper()
            ), f"Parameter `{name}` has invalid lower characters in dims string"
            if isinstance(data, Dict):
                data = pd.Series(data)

            if fill is not None:
                assert dims != "", f"Don't use `fill` argument with scalar entity {name}."
                data = pd.Series(data=fill, name=name, index=self._get_idx(name))

            if isinstance(data, (np.ndarray, list)):
                data = pd.Series(data=data, name=name, index=self._get_idx(name))

            assert isinstance(data, (pd.Series, pd.DataFrame)), (
                f"'{name}' has no trailing underscore in the name. "
                f"So it indicates a non-scalar entity but is a {type(data)}-type."
            )

        if not update:
            self._warn_if_unexpected_unit(name, unit)
            self.params._meta[name] = dict(
                doc=doc, unit=unit, src=src, etype=etype, comp=comp, dims=dims
            )

        if isinstance(data, pd.Series):
            data.rename(name, inplace=True)

        if hp.is_time_series_with_one_dimension_less_than_the_name(data, name):
            data = hp.from_T_to_simple_KG_format(data)
            assert hp.get_nDims(data=data) == hp.get_nDims(ent_name=name)

        if set_param:
            setattr(self.params, name, data)
            self.params._changed_since_last_dic_export = True
        return data

    def _warn_if_unexpected_unit(self, name, unit) -> None:
        etype = hp.get_etype(name)
        try:
            expected_units = getattr(Etypes, etype).units
        except AttributeError:
            pass
        else:
            if expected_units is None:
                return
            adder = "one of " if len(expected_units) > 1 else ""
            if expected_units == ():
                expected_units = "None"
            if not unit in expected_units:
                logger.warning(
                    f"Unexpected unit {unit} for entity {name}. Expected {adder}{expected_units}."
                )

    def collector(self, name: str, doc: str = "", unit: str = "") -> None:
        """Add a collector to the scenario."""
        setattr(self.collectors, name, dict())
        self.collectors._meta[name] = dict(doc=doc, unit=unit)

    def dim(
        self,
        name: str,
        data: Union[List, np.ndarray] = None,
        doc: str = "",
        unit: str = "",
        update=False,
    ) -> Union[List, np.ndarray]:
        """Add a dimension with coordinates to the scenario. Name must be a single capital letter."""

        assert len(name) == 1, f"Dimension name must be a single capital letter. '{name}' is given."
        assert data is not None, f"No data provided for dimension {name}."
        if not update:
            self.dims._meta[name] = dict(doc=doc, unit=unit)
        if isinstance(data[0], str):
            if data[0].isnumeric():
                logger.warn("Please do not give numeric data as strings, as it leads to errors")
        setattr(self.dims, name, data)
        return data

    def print_ents(self, filter_str: str = None, only_header=True) -> None:
        """Prints information about all parameters and variables that contain the filter string."""
        filter_addon = f" containing '{filter_str}'" if filter_str is not None else ""
        header = f"Entities{filter_addon} in scenario {self.id}"
        print(hp.bordered(header))

        for ent_name in self._all_ents_dict:
            if filter_str is None or filter_str in ent_name:
                ent_info = self.get_ent_info(ent_name, only_header=only_header)
                print(self._add_entity_type_prefix(ent_name, ent_info))

    def get_ent_info(self, ent_name: str, only_header: bool = True, show_units: bool = True) -> str:
        """Get a printable string with concise information of an entity."""
        ent_value = self.get_entity(ent_name)
        dim = hp.get_dims(ent_name)
        unit = self.get_unit(ent_name)

        if dim == "":
            unit_0d = f" {unit}" if show_units else ""
            string = f"{ent_name} = {ent_value}{unit_0d}\n"

        else:
            unit_nd = f" ({unit})" if show_units else ""
            if only_header and isinstance(ent_value, pd.Series):
                data = ent_value.head(4)

            else:
                data = ent_value

            string = f"{ent_name}{unit_nd}:\n{data}\n"

        return string

    def _add_entity_type_prefix(self, ent_name: str, string: str) -> str:
        """Adds p or v prefix to a a string."""
        ent_type = self._get_entity_type(ent_name)
        return f"{ent_type}.{string}"

    def get_CAP(self, which="CAPn", agg: bool = False) -> Dict[str, Union[float, pd.Series]]:
        """Returns a dictionary with the new or existing capacities.

        Args:
            which: One of 'CAPn' or 'CAPx'.
            agg: If True, multi-dimensional CAP entities are aggregated.
        """
        d = dict()

        assert which in ("CAPn", "CAPx"), "`which` need to be either CAPn or CAPx."
        container = self.res if which == "CAPn" else self.params

        for n, v in container.get_all().items():
            if which in n:
                if agg and isinstance(v, pd.Series):
                    v = v.sum()
                d[hp.get_component(n)] = v
        return d

    def get_EG_full_load_hours(self) -> float:
        """Returns the full load hours of the electricity purchase from the grid."""
        return (
            self.res.P_EG_buy_T.sum()
            * self.params.k__dT_
            * self.params.k__PartYearComp_
            / self.res.P_EG_buyPeak_
        )

    def get_all_collector_values(self, cache: bool = True) -> Dict[str, Dict[str, float]]:
        if not cache or not hasattr(self, "collector_values"):
            self._cache_collector_values()
        return getattr(self, "collector_values")

    def _cache_collector_values(self) -> None:
        d = {k: self.get_collectorValues(bal_name=k) for k in self.collectors.get_all()}
        setattr(self, "collector_values", d)

    def get_collectorValues(self, bal_name: str) -> Dict[str, float]:
        collector = getattr(self.collectors, bal_name)
        return {comp: self._get_BalTermValues(bal_name, term) for comp, term in collector.items()}

    def _get_BalTermValues(self, bal_name: str, term: Any) -> float:

        if callable(term):

            if "KG" == hp.get_dims(bal_name)[:2]:

                def weight(value, i):
                    return value * self.dt(k=i[0], g=i[1]) * self.periodOccurrences[i[0]]

            else:

                def weight(value, i):
                    return value

            idx = self._get_idx(bal_name)
            if isinstance(idx, pd.MultiIndex):
                values = [weight(hp.get_value_from_varOrPar(term(*i)), i) for i in idx]
            else:
                values = [weight(hp.get_value_from_varOrPar(term(i)), i) for i in idx]
            return sum(values)
        else:
            return hp.get_value_from_varOrPar(term)

    def make_sankey_string_from_collectors(self) -> str:
        templates = {
            "P_EL_source_KG": "E {k} el_hub {v}",
            "P_EL_sink_KG": "E el_hub {k} {v}",
            "dQ_cooling_source_KGN": "C {k} cool_hub {v}",
            "dQ_cooling_sink_KGN": "C cool_hub {k} {v}",
            "dQ_heating_source_KGH": "Q {k} heat_hub {v}",
            "dQ_heating_sink_KGH": "Q heat_hub {k} {v}",
            "F_fuel_F": "F FUEL {k} {v}",
            "dQ_amb_source_": "M {k} ambient {v}",
            "dQ_amb_sink_": "M ambient {k} {v}",
            "dH_hydrogen_source_KG": "H {k} H<sub>2</sub> {v}",
            "dH_hydrogen_sink_KG": "H H<sub>2</sub> {k} {v}",
        }
        header = ["type source target value"]
        rows = [
            templates[name].format(k=k, v=v)
            for name, collector in self.collector_values.items()
            for k, v in collector.items()
            if name in templates
        ]
        return "\n".join(header + rows)

    def _get_flat_col_names(self, name: str, df: pd.DataFrame) -> List:
        l = []
        for x in df.columns.to_flat_index():
            if isinstance(x, tuple):
                x = ", ".join(map(str, x))

            x = f"[{x}]"
            l.append(f"{name}{x}")
        return l

    def _get_flat_df_of_one_entity(
        self, ent_name: str, data: pd.Series, n_time_dims: int
    ) -> pd.DataFrame:
        data.name = ent_name
        dims = hp.get_dims(ent_name)
        if len(dims) == n_time_dims:
            df = data.to_frame()
        else:
            df = data.unstack(level=list(range(n_time_dims, len(dims))))
            df.columns = self._get_flat_col_names(ent_name, df)
        return df

    def get_flat_T_df(self, name_cond: Optional[Callable] = None, dims: str = "T") -> pd.DataFrame:
        """Get a Dataframe with all time-dependent entities. Additional dimensions are flattened.

        Args:
            name_cond: A function that takes the entity name and returns a True if the
                entity should be kept.
            dims: A string with the relevant time dimensions, e.g., 'T' or 'KG'.
        """
        return pd.concat(
            [
                self._get_flat_df_of_one_entity(n, ser, len(dims))
                for n, ser in self.yield_all_ents()
                if (dims in hp.get_dims(n) and (True if name_cond is None else name_cond(n)))
            ],
            axis=1,
        )

    @property
    def has_TSA(self):
        """If time series aggregation was conducted on the scenario."""
        return hasattr(self, "segmentation")

    def aggregate_temporally(
        self,
        n_typical_periods: int = 10,
        n_steps_per_period: int = 24,
        segmentation: bool = True,
        n_segments_per_period: int = 5,
        cluster_method: str = "hierarchical",
        sort_values: bool = False,
        store_TSA_instance: bool = False,
        weight_dict: Optional[Dict] = None,
        **kwargs,
    ) -> Scenario:
        # CREDITS: This function is heavily based on code from
        # https://github.com/FZJ-IEK3-VSA/FINE (MIT licence)
        """Do a time series aggregation (TSA) of all components considered in the energy
        system model.
        For the TSA itself, the tsam package (https://github.com/FZJ-IEK3-VSA/tsam) is used.
        Please refer to the tsam documentation for more information.

        Args:
            n_typical_periods: (`|K|`) Number of typical periods into which the time series
                data should be clustered. The number of time steps per period must be an integer
                multiple of the total number of considered time steps in the energy system.
            n_steps_per_period: Number of time steps per period
            segmentation: If the typical periods should be further segmented to fewer time steps.
            n_segments_per_period: (`|G|`) Number of segments per period.
            cluster_method: Method which is used in the tsam package for TSA the
                time series data. Examples: 'averaging', 'k_means', 'exact k_medoid' or
                'hierarchical'.
            sort_values: If the algorithm in the tsam package should use
                (a) the sorted duration curves (-> True) or
                (b) the original profiles (-> False) of the time series
                data within a period for TSA.
            store_TSA_instance: If the TimeSeriesAggregation instance created during TSA
                should be stored in the Scenario instance.
            weight_dict: Dictionary which weights the profiles. It is done by scaling
                the time series while the normalization process. Normally all time
                series have a scale from 0 to 1. By scaling them, the values get
                different distances to each other and with this, they are
                differently evaluated while the clustering process.
            kwargs: Additional keyword arguments for the TimeSeriesAggregation instance, e.g.,
                add extreme periods to the clustered typical periods.
        """

        hp.check_TSA_input(n_typical_periods, n_steps_per_period, len(self.dtindex_custom))
        if len(self.dtindex_custom) < len(self.dtindex) and n_typical_periods != 1:
            logger.warn("You should not call `set_time_horizon` before temporal aggregation.")

        if segmentation:
            if n_segments_per_period > n_steps_per_period:
                logger.warn(
                    "The chosen number of segments per period exceeds the number of time steps"
                    " per period. The number of segments per period is set to the number of time"
                    " steps per period."
                )
                n_segments_per_period = n_steps_per_period

        self._set_time_trace()
        logger.info(
            f"\nClustering time series data with {n_typical_periods} typical periods and"
            f" {n_steps_per_period} time steps per period"
            + f"\nfurther clustered to {n_segments_per_period} segments per period."
            if segmentation
            else "."
        )

        common_TSA_kwargs = dict(
            timeSeries=self.get_flat_T_df(dims="KG").set_axis(self.dtindex_custom),
            noTypicalPeriods=n_typical_periods,
            clusterMethod=cluster_method,
            sortValues=sort_values,
            weightDict=weight_dict,
            hoursPerPeriod=int(n_steps_per_period * self.step_width),
            segmentation=segmentation,
            **kwargs,
        )

        if segmentation:
            agg = TimeSeriesAggregation(noSegments=n_segments_per_period, **common_TSA_kwargs)
            data = pd.DataFrame(agg.clusterPeriodDict).reset_index(level=2, drop=True)
            timeStepsPerSegment = pd.DataFrame(agg.segmentDurationDict)["Segment Duration"]
        else:
            agg = TimeSeriesAggregation(**common_TSA_kwargs)
            data = pd.DataFrame(agg.clusterPeriodDict)

        data = data.rename_axis(list("KG"))
        self.update_params(**stack_data_from_TSA(data))

        if segmentation:
            self.segmentsPerPeriod = list(range(n_segments_per_period))
            self.timeStepsPerSegment = timeStepsPerSegment
            self.hoursPerSegment = self.step_width * self.timeStepsPerSegment
            # Define start time hour of each segment in each typical period:
            sst = self.segmentStartTime = self.hoursPerSegment.groupby(level=0).cumsum()
            sst.index = sst.index.set_levels(sst.index.levels[1] + 1, level=1)
            lvl0, lvl1 = sst.index.levels
            sst = sst.reindex(pd.MultiIndex.from_product([lvl0, [0, *lvl1]]))
            sst[sst.index.get_level_values(1) == 0] = 0

        # Bools
        self.segmentation = segmentation

        # Pandas DataFrame
        self.tsaAccuracyIndicators = pd.concat(
            [agg.accuracyIndicators(), agg.totalAccuracyIndicators().rename("Total").to_frame().T],
            axis=0,
        )

        # Lists
        self.typicalPeriods = list(agg.clusterPeriodIdx)
        self.timeStepsPerPeriod = list(range(n_steps_per_period))
        n_periods = int(len(self.dtindex) / n_steps_per_period)
        self.periods = list(range(n_periods))
        self.interPeriodTimeSteps = list(range(n_periods + 1))
        self.periodsOrder = agg.clusterOrder
        self.periodOccurrences = [(self.periodsOrder == tp).sum() for tp in self.typicalPeriods]

        # TSA object
        if store_TSA_instance:
            self.tsaInstance = agg

        # Model params
        self.dim("K", data=self.typicalPeriods, update=True)
        self.dim(
            "G",
            data=list(self.segmentsPerPeriod if segmentation else self.timeStepsPerPeriod),
            update=True,
        )
        self.dim("I", data=self.periods, update=True)
        self.param(
            "k__tsaComplexityReduction_",
            data=len(self.dims.K) * len(self.dims.G) / len(self.dtindex),
            doc="Factor to which the total time steps are reduced to during TSA",
        )
        self.param("n__typPeriods_", data=len(self.dims.K), doc="Number of typical periods")
        self.update_params(n__stepsPerPeriod_=n_steps_per_period)
        self.param(
            "n__segmentsPerPeriod_",
            data=len(self.dims.G),
            doc="Number of segments per typical period",
        )
        self._update_time_param(
            "t__TSA_", "Time to run time series aggregation", self._get_time_diff()
        )

        return self


def turn_numeric_strings_into_int(data):
    try:
        return int(data)
    except ValueError:
        pass
    return data


def stack_data_from_TSA(data: pd.DataFrame) -> Dict:
    d: Dict = dict()
    for k, v in data.items():
        if "[" in k:
            splitter = k.split("[")
            ent_name = splitter[0]
            idx = splitter[1][:-1].split(", ")
            # now, idx may contain numeric strings, e.g. ["1", "3", "spam"] or ["1"]
            my_idx = (
                tuple(turn_numeric_strings_into_int(i) for i in idx)
                if len(idx) > 1
                else turn_numeric_strings_into_int(idx[0])
            )
            if ent_name in d:
                d[ent_name][my_idx] = v
            else:
                d[ent_name] = {my_idx: v}
            assert isinstance(d[ent_name], Dict)
        else:
            d[k] = v
            assert isinstance(d[k], pd.Series)
    for k, v in d.items():
        if isinstance(v, Dict):
            df = pd.DataFrame(v)
            ser = df.stack(list(range(df.columns.nlevels)))
            v = ser.rename(k)
        elif isinstance(v, pd.DataFrame):
            v = v.squeeze()
        d[k] = v
    return d


def data_contains_nan(data: Optional[Union[int, float, list, np.ndarray, pd.Series]]) -> bool:
    if isinstance(data, float):
        return math.isnan(data)
    elif isinstance(data, list):
        for i in data:
            if math.isnan(i):
                return True
        return False
    elif isinstance(data, np.ndarray):
        return np.isnan(np.sum(data))
    elif isinstance(data, pd.Series):
        return data.isnull().values.any()
    else:
        return False


def warn_if_data_contains_nan(
    data: Optional[Union[int, float, list, np.ndarray, pd.Series]], name: str
) -> None:
    if data_contains_nan(data):
        logger.warning(f"Parameter '{name}' contains one or more NaNs.")
