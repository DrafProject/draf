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

from draf import helper as hp
from draf import paths
from draf.conventions import Etypes
from draf.core.datetime_handler import DateTimeHandler
from draf.core.draf_base_class import DrafBaseClass
from draf.core.entity_stores import Balances, Dimensions, Params, Results, Vars
from draf.core.mappings import GRB_OPT_STATUS, VAR_PAR
from draf.core.time_series_prepper import TimeSeriesPrepper
from draf.model_builder.abstract_component import Component
from draf.plotting import ScenPlotter
from draf.prep.data_base import ParDat
from draf.tsa.demand_analyzer import DemandAnalyzer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class Scenario(DrafBaseClass, DateTimeHandler):
    """An energy system configuration scenario.

    Args:
        id: Scenario Id string which must not start with a number.
        name: Scenario name string.
        doc: Scenario documentation string.
    """

    def __init__(
        self,
        freq: str,
        year: int,
        country: str = "DE",
        dtindex: Optional[int] = None,
        dtindex_custom: Optional[int] = None,
        t1: Optional[int] = None,
        t2: Optional[int] = None,
        id: str = "",
        name: str = "",
        doc: str = "",
        coords: Optional[Tuple[float, float]] = None,
        cs_name: str = "no_case_study",
        components: Optional[List[Union[Component, type]]] = None,
        consider_invest: bool = False,
    ):
        self.id = id
        self.name = name
        self.doc = doc
        self.country = country
        self.consider_invest = consider_invest
        self.coords = coords
        self.cs_name = cs_name
        self.mdl_language = "gp"

        self.dims = Dimensions()
        self.params = Params()
        self.plot = ScenPlotter(sc=self)
        self.prep = TimeSeriesPrepper(sc=self)
        self.vars = Vars()
        self.balances = Balances()

        if dtindex is None and dtindex_custom is None and t1 is None and t2 is None:
            self._set_dtindex(year=year, freq=freq)
        else:
            self.year = year
            self.dtindex = dtindex
            self.dtindex_custom = dtindex_custom
            self._t1 = t1
            self._t2 = t2
            self.freq = freq

        self.dim("T", infer=True)
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
            self.add_components(components)

    def __getstate__(self) -> Dict:
        """Remove objects with dependencies for serialization with pickle."""
        state = self.__dict__.copy()
        state.pop("mdl", None)
        state.pop("plot", None)
        state.pop("prep", None)
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
                "balances",
                "components",
                "res",
            ]
        )

    @property
    def size(self):
        return hp.human_readable_size(hp.get_size(self))

    def info(self):
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

    def _set_time_trace(self):
        self._time = time.perf_counter()

    def _get_time_diff(self):
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
        if hasattr(self, "res"):
            d.update(self.res.get_all())
        return d

    @property
    def has_thermal_entities(self) -> bool:
        return len(self.params.filtered(etype="dQ")) != 0

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
            etype: The component part of the entity name
            comp: The component part of the entity name, e.g. `BES`
            desc: The description part ot the entity name, e.g. `in`
            dims: The dimension part of the entity name, e.g. `T`
            func: Function that takes the entity name and returns a if the entity
                should be included or not
            params: If parameters are included
            vars: If variable results are included
        """
        kwargs = dict(etype=etype, comp=comp, desc=desc, dims=dims, func=func)
        d = dict()
        if params:
            d.update(self.params.filtered(**kwargs))
        if hasattr(self, "res") and vars:
            d.update(self.res.filtered(**kwargs))
        return d

    def get_mdpv(self) -> Tuple[gp.Model, Dimensions, Params, Vars]:
        return (self.mdl, self.dims, self.params, self.vars)

    def _set_default_solver_params(self) -> None:
        defaults = {
            "LogFile": str(self._res_fp / "gurobi.log"),
            "LogToConsole": 1,
            "OutputFlag": 1,
            "MIPGap": 0.1,
            "MIPFocus": 1,
        }

        for param, value in defaults.items():
            self.mdl.setParam(param, value, verbose=False)

    def update_par_dic(self):
        self._par_dic = self.params._to_dims_dic()

    def get_total_energy(self, data: pd.Series) -> float:
        return data.sum() * self.step_width

    gte = get_total_energy

    @property
    def par_dic(self):
        """Creates the par_dic at the first use then caches it. Use `update_par_dic()` to update."""
        if not hasattr(self, "_par_dic") or self.params._changed_since_last_dic_export:
            self._par_dic = self.params._to_dims_dic()
        return self._par_dic

    def update_res_dic(self):
        self._res_dic = self.res._to_dims_dic()

    def analyze_demand(self, data: pd.Series) -> DemandAnalyzer:
        da = DemandAnalyzer(p_el=data, year=self.year, freq=self.freq)
        da.show_stats()
        return da

    @property
    def _is_optimal(self):

        if self.mdl_language == "gp":
            return self.mdl.Status == gp.GRB.OPTIMAL

        elif self.mdl_language == "pyo":
            return self._termination_condition == pyo.TerminationCondition.optimal

        else:
            RuntimeError("`mdl_language` must be 'gp' or 'pyo'.")

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

    def get_entity(self, ent: str) -> Union[float, pd.Series]:
        """Get entity-data by its name."""
        return self._all_ents_dict[ent]

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

    def add_components(self, components: List):
        self._set_time_trace()
        logger.info(f"Set params for scenario {self.id}")
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
        mdl_language: str = "gp",
    ) -> Scenario:
        """Instantiates an optimization model and sets the custom_model_func on top of the given
        parameters and meta-informations for variables.

        Args:
            custom_model_func: A functions that takes the arguments sc, m, d, p, v and populates
                the parameters and variable meta data.
            custom_model_func_loc: Location of the custom_model_func. Determines when given
                custom_model_func is executed compared to the model_funcs of the components.
            speed_up: If speed increases should be exploited by converting the parameter objects to
                tuple-dicts before building the constraints.
            mdl_language: Choose either 'gp' or 'pyo'.
        """
        self.mdl_language = mdl_language

        # TODO: The "factory" design pattern may be suitable to cover _instantiate_model and
        #       activate_vars (https://refactoring.guru/design-patterns/factory-method)
        self._instantiate_model()

        self._set_time_trace()
        self._activate_vars()
        self._update_time_param("t__vars_", "Time to activate variables", self._get_time_diff())

        self._set_time_trace()

        params = self.params
        if speed_up and self.mdl_language == "gp":
            params = self.get_tuple_dict_container(params)

        logger.info(f"Set model for scenario {self.id}.")

        model_func_list = []
        if hasattr(self, "components"):
            model_func_list += [comp.model_func for comp in self.components]
        if custom_model_func is not None:
            model_func_list.insert(custom_model_func_loc, custom_model_func)

        for model_func in model_func_list:
            model_func(sc=self, m=self.mdl, d=self.dims, p=params, v=self.vars)

        self._update_time_param("t__model_", "Time to build model", self._get_time_diff())
        return self

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

    def _instantiate_model(self):

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

    def _activate_gurobipy_vars(self):
        for name, metas in self.vars._meta.items():

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

    def _activate_pyomo_vars(self):
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
    ) -> Scenario:
        """Solves the optimization problem and does postprocessing if the function is given.
        Results are stored in the Results-object of the scenario.

        Args:
            logToConsole: Sets the LogToConsole param to the gurobipy model.
            outputFlag: Sets the outputFlag param to the gurobipy model.
            show_results: If the Cost and Carbon emissions are printed.
            keep_vars: If the variable objects are kept after optimization run.
            postprocess_func: Function which is executed with the results container object as
                argument.
        """
        for k, v in self.params.get_all().items():
            warn_if_data_contains_nan(data=v, name=k)

        if not hasattr(self, "mdl"):
            self.set_model()

        pp_funcs = []
        if postprocess_func is not None:
            pp_funcs.append(postprocess_func)
        if hasattr(self, "components"):
            for comp in self.components:
                if hasattr(comp, "postprocess_func"):
                    pp_funcs.append(comp.postprocess_func)

        kwargs = dict(
            logToConsole=logToConsole,
            outputFlag=outputFlag,
            show_results=show_results,
            keep_vars=keep_vars,
            postprocess_funcs=pp_funcs,
        )

        if self.mdl_language == "gp":
            assert which_solver == "gurobi"
            self._optimize_gurobipy(**kwargs)

        elif self.mdl_language == "pyo":
            self._optimize_pyomo(**kwargs, which_solver=which_solver)

        if hasattr(self, "balances"):
            self._cache_balance_values()
        return self

    def _optimize_gurobipy(
        self, logToConsole, outputFlag, show_results, keep_vars, postprocess_funcs
    ):
        logger.info(f"Optimize {self.id}")
        self._set_time_trace()
        self.mdl.setParam("LogToConsole", int(logToConsole), verbose=False)
        self.mdl.setParam("OutputFlag", int(outputFlag), verbose=False)
        self.mdl.optimize()
        self._update_time_param("t__solve_", "Time to solve the model", self._get_time_diff())
        status = self.mdl.Status
        status_str = GRB_OPT_STATUS[status]
        if status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
            self.res = Results(self)
            if not keep_vars:
                del self.vars
            for ppf in postprocess_funcs:
                ppf(self.res)
            if status == gp.GRB.TIME_LIMIT:
                logger.warning("Time-limit reached")
            if show_results:
                try:
                    print(
                        f"{self.id}: C_TOT_={self.res.C_TOT_:.2f}, "
                        f"CE_TOT_={self.res.CE_TOT_:.2f} ({status_str})"
                    )
                except ValueError:
                    logger.warning("res.C_TOT_ or res.CE_TOT_ not found.")
        else:
            if status in [gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE]:
                response = input(
                    "The model is infeasible. Do you want to compute an Irreducible"
                    " Inconsistent Subsystem (IIS)?\n[(y)es / (n)o] + ENTER"
                )
                if response == "y":
                    self.calculate_IIS()
            raise RuntimeError(
                f"ERROR solving scenario {self.name}: mdl.Status= {status} ({GRB_OPT_STATUS[status]}) --> {self.mdl.Params.LogFile}"
            )

    def _optimize_pyomo(
        self, logToConsole, outputFlag, show_results, keep_vars, postprocess_funcs, which_solver
    ):
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
            if not keep_vars:
                del self.vars
            for ppf in postprocess_funcs:
                ppf(self.res)
            if tc == pyo.TerminationCondition.maxTimeLimit:
                logger.warning("Time-limit reached")
            if show_results:
                try:
                    print(
                        f"{self.id}: C_TOT_={self.res.C_TOT_:.2f}, CE_TOT_={self.res.CE_TOT_:.2f} ({tc})"
                    )
                except ValueError:
                    logger.warning("res.C_TOT_ or res.CE_TOT_ not found.")
        else:
            raise RuntimeError(
                f"ERROR solving scenario {self.name}: status= {status}, ",
                f"termination condition={tc}) --> logfile: {logfile}",
            )

    def calculate_IIS(self):
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
        given, the file is saved in the case study's default result directory."""
        date_time = self._get_now_string()
        if fp is None:
            fp = self._res_fp / f"{date_time}_{self.id}.{filetype}"

        self.mdl.write(str(fp))
        logger.info(f"written to {fp}")

    def save(self) -> None:
        """Saves the scenario to a pickle-file."""
        date_time = self._get_now_string()

        fp = self._res_fp / f"{date_time}_{self.id}.p"

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

    def _infer_dimension_from_name(self, name: str) -> Tuple[str, str, Union[float, pd.Series]]:
        if name == "T":
            doc = f"{self.freq} time steps"
            unit = self.freq_unit
            data = list(range(self._t1, self._t2 + 1))
        else:
            raise AttributeError(f"No infer options available for {name}")
        return doc, unit, data

    def _get_idx(self, ent_name: str) -> Union[List, pd.MultiIndex]:
        dims = hp.get_dims(ent_name)
        coords = self.get_coords(dims)
        if len(dims) == 1:
            idx = coords[0]
        else:
            idx = pd.MultiIndex.from_product(coords, names=list(dims))
        return idx

    def get_unit(self, ent_name: str) -> Optional[str]:
        return self.get_meta(ent_name=ent_name, meta_type="unit")

    def get_doc(self, ent_name: str) -> Optional[str]:
        return self.get_meta(ent_name=ent_name, meta_type="doc")

    def get_src(self, ent_name: str) -> Optional[str]:
        return self.get_meta(ent_name=ent_name, meta_type="src")

    def get_meta(self, ent_name: str, meta_type: str) -> Optional[str]:
        """Returns meta-information such as doc or unit for a given entity.

        Note:
            Meta-information are stored as followed:
            sc.res (obj)
            sc.res._meta (dict)
            sc.res._meta[<entity-name>] (dict with metas {"doc":..., "unit":...})
        """
        return_value = None
        for attr in ["params", "res", "dims", "balances"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                metas = obj._meta.get(ent_name, "")
                if metas != "":
                    return metas.get(meta_type, "")
        return None

    def update_params(self, **kwargs) -> Scenario:
        """Update multiple existing parameters.
        e.g. sc.update_params(P_EG_dem_T=2000, c_EG_addon_=0, c_EG_peak_=0)
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

    def param(
        self,
        name: Optional[str] = None,
        data: Optional[Union[int, float, list, np.ndarray, pd.Series]] = None,
        doc: str = "",
        unit: str = "",
        src: str = "",
        fill: Optional[float] = None,
        update: bool = False,
        from_db: Optional[ParDat] = None,
    ) -> pd.Series:
        """Add a parameter to the scenario.

        Args:
            name: The entity name. It has to end with an underscore followed by the
                single-character dimensions i.e. a solely time-dependent parameter has
                to end with `_T` e.g. `P_eDem_T`;
                a scalar has to end with `_` e.g. `C_TOT_inv_`.
            data: Data is normally given as int, float or pd.Series. Lists and np.ndarrays are
                converted to pd.Series.
            doc: A description string.
            fill: If a float is given here, for all relevant dimensions inferred from the name the
                series is filled.
            update: If True, the meta-data will not be touched, just the data changed.
            read_kwargs: Keyword arguments handed onto the pandas read-function.
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

        setattr(self.params, name, data)
        self.params._changed_since_last_dic_export = True
        return data

    def _warn_if_unexpected_unit(self, name, unit):
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
                    f"Unexpected unit {unit} for entity {name}. "
                    f"Expected {adder}{expected_units}."
                )

    def balance(
        self,
        name: str,
        doc: str = "",
        unit: str = "",
    ) -> None:
        """Add a balance to the scenario"""
        setattr(self.balances, name, dict())
        self.balances._meta[name] = dict(doc=doc, unit=unit)

    def dim(
        self,
        name: str,
        data: Union[List, np.ndarray] = None,
        doc: str = "",
        unit: str = "",
        infer=False,
    ) -> None:
        """Add a dimension with coordinates to the scenario. Name must be a single capital letter."""

        assert len(name) == 1, f"Dimension name must be a single capital letter. '{name}' is given."
        if infer:
            assert data is None, "if `infer=True`, then data must be None."
            doc, unit, data = self._infer_dimension_from_name(name)

        assert data is not None, f"No data provided for {name}. Infer with `infer=True`"
        self.dims._meta[name] = dict(doc=doc, unit=unit)
        setattr(self.dims, name, data)
        return data

    def print_ents(self, filter_str: str = None, only_header=True) -> None:
        """Prints informations about all parameters and variables that contain the filter string."""
        filter_addon = f" containing '{filter_str}'" if filter_str is not None else ""
        header = f"Entities{filter_addon} in scenario {self.id}"
        print(hp.bordered(header))

        for ent_name in self._all_ents_dict:
            if filter_str is None or filter_str in ent_name:
                ent_info = self.get_ent_info(ent_name, only_header=only_header)
                print(self._add_entity_type_prefix(ent_name, ent_info))

    def get_ent_info(self, ent_name: str, only_header: bool = True, show_units: bool = True) -> str:
        """Get an printable string with concise info of an entity."""
        ent_value = self.get_entity(ent_name)
        dim = hp.get_dims(ent_name)
        unit = self.get_unit(ent_name)

        if dim == "":
            unit_0d = f" {unit}" if show_units else ""
            string = f"{ent_name} = {ent_value}{unit_0d}\n"

        else:
            unit_nd = f" [{unit}]" if show_units else ""
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

    def get_EG_full_load_hours(self):
        return (
            self.res.P_EG_buy_T.sum()
            * self.params.k__dT_
            * self.params.k__PartYearComp_
            / self.res.P_EG_buyPeak_
        )

    def get_all_balance_values(self, cache: bool = True) -> Dict[str, Dict[str, float]]:
        if not cache or not hasattr(self, "balance_values"):
            self._cache_balance_values()
        return getattr(self, "balance_values")

    def _cache_balance_values(self) -> None:
        d = {k: self.get_balanceValues(bal_name=k) for k in self.balances.get_all()}
        setattr(self, "balance_values", d)

    def get_balanceValues(self, bal_name: str) -> Dict[str, float]:
        balance = getattr(self.balances, bal_name)
        return {comp: self._get_BalTermValues(bal_name, term) for comp, term in balance.items()}

    def _get_BalTermValues(self, bal_name: str, term: Any) -> float:
        if hp.is_a_lambda(term):
            idx = self._get_idx(bal_name)
            if isinstance(idx, pd.MultiIndex):
                return sum(hp.get_value_from_varOrPar(term(*i)) for i in idx)
            else:
                return sum(hp.get_value_from_varOrPar(term(i)) for i in idx)
        else:
            return hp.get_value_from_varOrPar(term)

    def make_sankey_string_from_balances(self):
        templates = {
            "P_EL_source_T": "E {k} el_hub {v}",
            "P_EL_sink_T": "E el_hub {k} {v}",
            "dQ_cooling_source_TN": "Q {k} cool_hub {v}",
            "dQ_cooling_sink_TN": "Q cool_hub {k} {v}",
            "dQ_heating_source_TH": "Q {k} heat_hub {v}",
            "dQ_heating_sink_TH": "Q heat_hub {k} {v}",
            "F_fuel_F": "F FUEL {k} {v}",
            "dQ_amb_source_": "Q {k} ambient {v}",
            "dQ_amb_sink_": "Q ambient {k} {v}",
        }
        header = ["type source target value"]
        rows = [
            templates[name].format(k=k, v=v)
            for name, balance in self.get_all_balance_values().items()
            for k, v in balance.items()
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

    def _get_flat_df_of_one_entity(self, ent_name: str, data: pd.Series) -> pd.DataFrame:
        dim = hp.get_dims(ent_name)
        if len(dim) == 1:
            df = data.to_frame()
        else:
            df = data.unstack(level=list(range(1, len(dim))))
            df.columns = self._get_flat_col_names(ent_name, df)
        return df

    def get_flat_T_df(self, name_cond: Optional[Callable] = None) -> pd.DataFrame:
        """Get a Dataframe with all time-dependent entities. Additional dimensions are flattened.
        Args:
            name_cond: A function that takes the entity name and returns a True if the
                entity should be kept.
        """

        def cond(n):
            c = "T" in hp.get_dims(n)

            if name_cond is not None:
                c = c and name_cond(n)

            return c

        l = [self._get_flat_df_of_one_entity(n, ser) for n, ser in self.yield_all_ents() if cond(n)]
        return pd.concat(l, 1)


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


def warn_if_data_contains_nan(
    data: Optional[Union[int, float, list, np.ndarray, pd.Series]], name: str
) -> None:
    if data_contains_nan(data):
        logger.warning(f"Parameter '{name}' contains one or more NaNs.")
