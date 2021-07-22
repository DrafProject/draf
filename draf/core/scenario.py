from __future__ import annotations

import copy
import datetime
import logging
import math
import pickle
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gurobipy as gp
import numpy as np
import pandas as pd
from draf import helper as hp
from draf.core.draf_base_class import DrafBaseClass
from draf.core.entity_stores import Dimensions, Params, Results, Vars
from draf.core.mappings import GRB_OPT_STATUS, VAR_PAR
from draf.plotting import ScenPlotter
from draf.prep.params_prepping import Prepper

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


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


class Scenario(DrafBaseClass):
    """An energy system configuration scenario.

    Args:
        cs: CaseStudy object.
        id: Scenario Id string which must not start with a number.
        name: Scenario name string.
        doc: Scenario documentation string.
    """

    def __init__(self, cs: "CaseStudy", id: str = "", name: str = "", doc: str = ""):
        self._cs = cs
        self.id = id
        self.name = name
        self.doc = doc
        self.dims = Dimensions()
        self.params = Params()
        self.mdl = None
        self.plot = ScenPlotter(sc=self)
        self.prep = Prepper(sc=self)
        self.vars = Vars()
        self.year = cs.year
        self.country = cs.country
        self.freq = cs.freq
        self.dtindex = cs.dtindex
        self.dtindex_custom = cs.dtindex_custom
        self._freq_unit = cs._freq_unit
        self._t1 = cs._t1
        self._t2 = cs._t2

    def __repr__(self):
        """Get overview of attributes of the scenario object."""
        preface = "<{} object>".format(self.__class__.__name__)
        attribute_list = []
        excluded = ["dims", "params", "vars", "dtindex", "dtindex_custom", "res"]
        for k, v in self.get_all().items():
            if k in excluded:
                v = "[...]"
            attribute_list.append(f"• {k}: {v}")
        return "{}\n{}".format(preface, "\n".join(attribute_list))

    def __getstate__(self) -> Dict:
        """Remove objects with dependencies for serialization with pickle."""
        d = self.__dict__.copy()
        d.pop("mdl", None)
        d.pop("_cs", None)
        return d

    @property
    def _all_ents_dict(self) -> Dict:
        """Returns a name:data Dict of all entities without meta data."""
        if hasattr(self, "res"):
            d = {**self.params.__dict__, **self.res.__dict__}
        else:
            d = self.params.__dict__
        d.pop("_meta", None)
        return d

    def set_default_solver_params(self) -> None:
        defaults = {
            "LogFile": str(self._cs._res_fp / "gurobi.log"),
            "LogToConsole": 1,
            "OutputFlag": 1,
            "MIPGap": 0.1,
            "MIPFocus": 1,
        }

        for param, value in defaults.items():
            self.mdl.setParam(param, value, verbose=False)

    def update_par_dic(self):
        self._par_dic = self.params._to_dims_dic()

    @property
    def par_dic(self):
        """Creates the par_dic at the first use then caches it. Use `update_par_dic()` to update."""
        if not hasattr(self, "_par_dic") or self.params._changed_since_last_dic_export:
            self._par_dic = self.params._to_dims_dic()
        return self._par_dic

    def update_res_dic(self):
        self._res_dic = self.res._to_dims_dic()

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

    def get_entity(
        self, ent: str, include_params: bool = True, include_res: bool = True
    ) -> Union[float, pd.Series]:
        """Get entity-data by its name.

        Args:
            ent: Entity name.
            include_params: If params are considered.
            include_res: If variables results are considered.
        """
        if ent in self.params.get_all():
            return self.params.get(ent)

        elif hasattr(self, "res"):
            if ent in self.res.get_all():
                return self.res.get(ent)

        else:
            raise AttributeError(f"Entity {ent} not found.")

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
        self._cs._set_time_trace()

        try:
            params_builder_func(self)

            for k, v in self.params.get_all().items():
                warn_if_data_contains_nan(data=v, name=k)

        except RuntimeError as e:
            logger.error(e)

        self.add_par(
            "timelog_params_",
            self._cs._get_time_diff(),
            doc="Time for building the params",
            unit="seconds",
        )
        return self

    def set_model(self, model_builder_func: Callable, speed_up: bool = True) -> Scenario:
        """Instantiates a Gurobipy model and sets the model_builder_func on top of the given
        parameters and meta-informations for variables.

        Args:
            model_builder_func: A functions that takes the arguments m, d, p, v and populates the
                parameters and variable meta data.
            speed_up: If speed increases should be exploited by converting the parameter objects to
                tuple-dicts before building the constraints.
        """
        self._instantiate_model(mdl_language="gurobipy")

        self._cs._set_time_trace()
        self._activate_vars()
        self.add_par(
            "timelog_vars_",
            self._cs._get_time_diff(),
            doc="Time for building the variables",
            unit="seconds",
        )

        self._cs._set_time_trace()

        if speed_up:
            params = self.get_tuple_dict_container()
        else:
            params = self.params

        logger.info(f"Set model for scenario {self.id}.")
        model_builder_func(m=self.mdl, d=self.dims, p=params, v=self.vars)
        self.add_par(
            "timelog_model_",
            self._cs._get_time_diff(),
            doc="Time for building the model",
            unit="seconds",
        )
        return self

    def get_tuple_dict_container(self) -> Params:
        """Returns a copy of the params object where all Pandas Series objects are converted
         to gurobipy's tupledicts in order to speed up the execution of the model_builder_func.

        Meta data are not copied.
        """
        p = self.params
        td = Params()
        for name, obj in p.get_all().items():
            if isinstance(obj, pd.Series):
                data = gp.tupledict(obj.to_dict())
            else:
                data = obj
            setattr(td, name, data)
        return td

    def _instantiate_model(self, mdl_language: str = "gurobipy"):
        if mdl_language == "gurobipy":
            self.mdl = gp.Model(self.id)
            self.set_default_solver_params()

    def _activate_vars(self) -> Scenario:
        """Instantiate variables according to the meta-data in `vars._meta`."""
        mdl = self.mdl

        for name, metas in self.vars._meta.items():

            kwargs = dict(lb=metas["lb"], ub=metas["ub"], name=name, vtype=metas["vtype"])

            if metas["is_scalar"]:
                var_obj = mdl.addVar(**kwargs)

            else:
                dims = self._get_dims(ent=name)
                dims_list = self.get_coords(dims=dims)
                var_obj = mdl.addVars(*dims_list, **kwargs)

            setattr(self.vars, name, var_obj)
        return self

    def optimize(
        self,
        logToConsole: bool = False,
        outputFlag: bool = True,
        show_results: bool = False,
        keep_vars: bool = True,
        postprocess_func: Optional[Callable] = None,
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
        logger.info(f"Optimize {self.id}.")
        self._cs._set_time_trace()
        self.mdl.setParam("LogToConsole", int(logToConsole), verbose=False)
        self.mdl.setParam("OutputFlag", int(outputFlag), verbose=False)
        self.mdl.optimize()
        self.add_par(
            "timelog_solve_",
            self._cs._get_time_diff(),
            doc="Time for solving the model",
            unit="seconds",
        )
        status = self.mdl.Status
        status_str = GRB_OPT_STATUS[status]
        if status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
            self.res = Results(self)
            if not keep_vars:
                del self.vars

            if postprocess_func is not None:
                postprocess_func(self.res)

            if status == gp.GRB.TIME_LIMIT:
                logger.warning("Time-limit reached")

            if show_results:
                try:
                    print(f"{self.id}: C_={self.res.C_:.2f}, CE_={self.res.CE_:.2f} ({status_str})")
                except ValueError:
                    logger.warning("res.C_ or res.CE_ not found.")

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

        return self

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
        scen_copy._cs = self._cs
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
            fp = self._cs._res_fp / f"{date_time}_{self.id}.{filetype}"

        self.mdl.write(str(fp))
        logger.info(f"written to {fp}")

    def save(self) -> None:
        """Saves the scenario to a pickle-file."""
        date_time = self._get_now_string()

        fp = self._cs._res_fp / f"{date_time}_{self.id}.p"

        try:
            with open(fp, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"saved scenario to {fp}")

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

    def add_var(
        self,
        name: str,
        doc: str = "",
        unit: str = "",
        lb: float = 0.0,
        ub: float = 1e100,
        vtype: str = "C",
        dims: Optional[str] = None,
    ) -> None:
        """Add metadata of one or more variables to the scenario.

        Note:
            * This does not yet create a gurobipy-variable-object.
            * If dims == None, dims are inferred from name suffix.

        """

        if dims is None:
            dims = self._get_dims(name)
        is_scalar = dims == ""
        self.vars._meta[name] = dict(
            doc=doc, unit=unit, lb=lb, ub=ub, dims=dims, is_scalar=is_scalar, vtype=vtype
        )

    def _infer_dimension_from_name(self, name: str) -> (str, str, Union[float, pd.Series]):
        if name == "T":
            doc = f"{self._cs.freq} time steps"
            unit = self._cs._freq_unit
            data = self._cs.get_T()
        else:
            raise AttributeError(f"No infer options available for {name}")
        return doc, unit, data

    def _get_idx(self, ent_name: str):
        dims = self._get_dims(ent_name)
        coords = self.get_coords(dims)
        if len(dims) == 1:
            idx = coords[0]
        else:
            idx = pd.MultiIndex.from_product(coords, names=list(dims))
        return idx

    def get_unit(self, ent_name: str) -> str:
        return self.get_meta(ent_name=ent_name, meta_type="unit")

    def get_doc(self, ent_name: str) -> str:
        return self.get_meta(ent_name=ent_name, meta_type="doc")

    def get_meta(self, ent_name: str, meta_type: str) -> str:
        """Returns meta-information such as a description (doc) or the pysical unit (unit)
        for a given entity.

        Note:
            Meta-information are stored as followed:
            sc.res (obj)
            sc.res._meta (dict)
            sc.res._meta[<entity-name>] (dict with metas {"doc":..., "unit":...})
        """
        for attr in ["params", "res", "dims"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                metas = obj._meta.get(ent_name, "")
                if metas is not "":
                    return metas.get(meta_type, "")

    def update_params(self, **kwargs) -> Scenario:
        """Update multiple existing parameters.
        e.g. sc.update_params(E_GRID_dem_T=2000, c_GRID_addon_T=0, c_el_peak_=0)
        """
        for ent_name, data in kwargs.items():

            if isinstance(data, str):
                try:
                    data = self.params.get(data)
                except AttributeError as e:
                    raise e

            if not hasattr(self.params, ent_name):
                raise RuntimeError(f"The parameter {ent_name} you want to update does not exist.")

            if self.fits_convention(ent_name, data):
                self.add_par(ent_name, data=data, update=True)
            else:
                self.add_par(ent_name, fill=data, update=True)

        return self

    def fits_convention(self, ent_name: str, data: Union[int, float, pd.Series]) -> bool:
        """Decides if the data dimensions and the entity name match according to the
         naming-conventions.
        """
        # TODO: maybe also check for the number of dimensions?
        dims = self._get_dims(ent_name)
        match_a = dims == "" and isinstance(data, (int, float))
        match_b = dims != "" and isinstance(data, (pd.Series))
        return match_a or match_b

    def add_par(
        self,
        name: str,
        data: Optional[Union[int, float, list, np.ndarray, pd.Series]] = None,
        doc: str = "",
        unit: str = "",
        fill: Optional[float] = None,
        update: bool = False,
        fp: Optional[str] = None,
        **read_kwargs,
    ) -> pd.Series:
        """Add a parameter to the scenario.

        Args:
            name: the entity name. It has to end with an underscore followed by the
                single-character dimensions i.e. a solely time-dependent parameter has
                to end with `_T` e.g. `E_dem_T`;
                a scalar has to end with `_` e.g. `C_inv_`.
            data: data is normally given as int, float or pd.Series. Lists and np.ndarrays are
                converted to pd.Series.
            doc: a description string.
            fill: if a float is given here, for all relevant dimensions inferred from the name the
                series is filled.
            update: if True, the meta-data will not be touched, just the data changed.
            fp: a file path can be given to read a h5 or csv-file.
            read_kwargs: keyword arguments handed onto the pandas read-function.

        """

        if fp is not None:
            data = hp.read_array(fp=fp, asType="s", **read_kwargs)

        dims = self._get_dims(name)

        if dims == "":
            assert isinstance(data, (float, int)), (
                f"'{name}' has trailing underscore in the name. "
                f"So it indicates a scalar entity but is a {type(data)}-type."
            )
        else:
            if fill is not None:
                assert dims != "", (
                    f"fill works not for scalars as {name}."
                    f"Please use the data argument instead."
                )
                data = pd.Series(data=fill, name=name, index=self._get_idx(name))

            if isinstance(data, (np.ndarray, list)):
                data = pd.Series(data=data, name=name, index=self._get_idx(name))

            assert isinstance(data, (pd.Series, pd.DataFrame)), (
                f"'{name}' has no trailing underscore in the name. "
                f"So it indicates a non-scalar entity but is a {type(data)}-type."
            )

        if not update:
            self.params._meta[name] = dict(doc=doc, unit=unit, dims=dims)

        if isinstance(data, pd.Series):
            data.rename(name, inplace=True)

        setattr(self.params, name, data)
        self.params._changed_since_last_dic_export = True
        return data

    def add_dim(
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
        dim = self._get_dims(ent_name)
        unit = self.get_unit(ent_name)

        if dim == "":
            unit_0d = f" {unit}" if show_units else ""
            string = f"{ent_name} = {ent_value}{unit_0d}\n"

        else:
            unit_nd = f" [{unit}]" if show_units else ""
            if only_header:
                data = ent_value.head(4)

            else:
                data = ent_value

            string = f"{ent_name}{unit_nd}:\n{data}\n"

        return string

    def _add_entity_type_prefix(self, ent_name: str, string: str) -> str:
        """Adds p or v prefix to a a string."""
        ent_type = self._get_entity_type(ent_name)
        return f"{ent_type}.{string}"

    def get_CAP(self, which="CAPn", agg: bool = True) -> Dict[str, Union[float, pd.Series]]:
        """Returns a dictionary with the new or existing capacities.

        Args:
            which: One of 'CAPn' or 'CAPx'.
            agg: If True, multi-dimensional CAP entities are aggregated.
        """
        d = dict()
        _map = {"CAPn": "res", "CAPx": "params"}
        container = getattr(self, _map[which])

        for n, v in container.get_all().items():
            if which in n:
                if agg and isinstance(v, pd.Series):
                    v = v.sum()
                d[n.split("_")[1]] = v
        return d
