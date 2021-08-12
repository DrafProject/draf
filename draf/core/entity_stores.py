import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from draf import helper as hp
from draf.core.draf_base_class import DrafBaseClass

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


class Scenarios(DrafBaseClass):
    """Stores scenarios."""

    def __repr__(self):
        layout = "{bullet}{name:<10} | {doc}\n"
        return self._build_repr(layout, which_metadata=["doc"])


class Dimensions(DrafBaseClass):
    """Stores dimensions."""

    def __init__(self):
        self._meta: Dict[str, Dict] = {}

    def __repr__(self):
        layout = "{bullet}{name:<5} {unit:>10}   {doc}\n"
        return self._build_repr(layout, ["doc", "unit"])


class EntityStore(DrafBaseClass):
    def __repr__(self):
        layout = "{bullet}{name:<20}{dims:>5} {unit:>14} {doc}\n"
        return self._build_repr(layout, which_metadata=["unit", "doc", "dims"])

    def __init__(self):
        self._changed_since_last_dic_export: bool = False

    @property
    def _empty_dims_dic(self) -> Dict[str, List[str]]:
        """Returns an empty dimension dictionary of the shape
        {<dimension>: [<ent_name1>, <ent_name2>]}.
        """
        ents_list = self.get_all().keys()
        dims_dic = {ent: hp.get_dims(ent) for ent in ents_list}
        dims_set = set(dims_dic.values())
        _empty_dims_dic = {dims: [] for dims in dims_set}
        for ent, dims in dims_dic.items():
            _empty_dims_dic[dims].append(ent)
        return _empty_dims_dic

    def _to_dims_dic(self, unstack_to_first_dim: bool = False) -> Dict[str, Union[Dict, pd.Series]]:
        """Returns a dimension dictionary where nonscalar params are stored in dataframes per
        dimension.

        Args:
            unstack_to_first_dim: If MultiIndex are unstacked resulting in Dataframes with a normal
                Index.

        """
        dims_dic = self._empty_dims_dic.copy()

        for dim in dims_dic:

            if dim == "":
                dims_dic[dim] = {ent: self.get(ent) for ent in dims_dic[dim]}

            else:
                # use the index of the first entity of given dimension
                first_el_of_empty_dims_dic: str = dims_dic[dim][0]
                multi_idx = self.get(first_el_of_empty_dims_dic).index
                multi_idx.name = dim
                dims_dic[dim] = pd.DataFrame(index=multi_idx)

                for ent in self._empty_dims_dic[dim]:
                    dims_dic[dim][ent] = getattr(self, ent)

                if unstack_to_first_dim:
                    dims_dic[dim] = dims_dic[dim].unstack(level=list(range(1, len(dim))))

        self._changed_since_last_dic_export = False
        return dims_dic

    def filtered(
        self,
        type: Optional[str] = None,
        component: Optional[str] = None,
        acro: Optional[str] = None,
        dims: Optional[str] = None,
    ) -> Dict:
        return {
            k: v
            for k, v in self.get_all().items()
            if (
                (hp.get_type(k) == type or type is None)
                and (hp.get_component(k) == component or component is None)
                and (hp.get_acro(k) == acro or acro is None)
                and (hp.get_dims(k) == dims or dims is None)
            )
        }


class Params(EntityStore):
    """Stores parameters in a convenient way together with its functions."""

    def __init__(self):
        super().__init__()
        self._meta: Dict[str, Dict] = {}

    def _set_meta(self, ent_name: str, meta_type: str, value: str) -> None:
        self._meta.setdefault(ent_name, {})[meta_type] = value

    def _convert_unit(self, ent_name: str, conversion_factor: float, return_unit: str):
        par = self.get(ent_name)
        par *= conversion_factor
        self._set_meta(ent_name, "unit", return_unit)


class Vars(EntityStore):
    """Stores optimization variables."""

    def __init__(self):
        super().__init__()
        self._meta: Dict[str, Dict] = {}

    def __getstate__(self):
        return dict(_meta=self._meta)


class Results(EntityStore):
    """Stores results in a easy accessible way together with its functions."""

    def __init__(self, sc: "Scenario"):
        super().__init__()
        self._get_results_from_variables(sc=sc)

    # TODO: Move to _get_results_from_variables, _from_gurobipy, _from_pyomo to Scenario for
    # better type hinting

    def _get_results_from_variables(self, sc: "Scenario") -> None:
        if sc.mdl_language == "gp":
            self._from_gurobipy(sc)
        else:
            self._from_pyomo(sc)

        self._changed_since_last_dic_export = True
        self._meta = sc.vars._meta

    def _from_gurobipy(self, sc: "Scenario") -> None:
        for name, var in sc.vars.get_all().items():
            dims = hp.get_dims(name)
            if dims == "":
                data = var.x
            else:
                dic = sc.mdl.getAttr("x", var)
                data = pd.Series(dic, name=name)
                data.index = data.index.set_names(list(dims))

            setattr(self, name, data)

    def _from_pyomo(self, sc: "Scenario") -> None:
        for name, var in sc.vars.get_all().items():
            dims = hp.get_dims(name)
            if dims == "":
                data = var.value
            else:
                dic = {index: var[index].value for index in var}
                data = pd.Series(dic, name=name)
                data.index = data.index.set_names(list(dims))

            setattr(self, name, data)

    def _get_meta(self, ent_name: str, meta_type: str) -> Any:
        try:
            return self._meta[ent_name][meta_type]
        except KeyError:
            return None

    def _set_meta(self, ent_name: str, meta_type: str, value: str) -> None:
        self._meta.setdefault(ent_name, {})[meta_type] = value

    def _copy_meta(self, source_ent: str, target_ent: str, which_metas: List = None) -> None:
        if which_metas is None:
            which_metas = ["doc", "unit", "dims"]

        for meta_type in which_metas:
            self._set_meta(
                ent_name=target_ent,
                meta_type=meta_type,
                value=self._get_meta(ent_name=source_ent, meta_type=meta_type),
            )

    def make_pos_ent(self, source: str, target_neg: str = None, doc_target: str = None) -> None:
        """Makes entities positive.

        If a target-entity-name is given, the negative values are stored as positive values in a
        new entity.

        Args:
            source: Source entity.
            target_neg: Negative target entity.
            doc_target: The doc string of the target.
        """
        try:
            source_ser = self.get(source)

            if target_neg is not None:
                setattr(self, target_neg, -source_ser.where(cond=source_ser < 0, other=0))

                self._copy_meta(
                    source_ent=source, target_ent=target_neg, which_metas=["doc", "unit", "dims"]
                )

                if isinstance(doc_target, str):
                    self._set_meta(target_neg, meta_type="doc", value=doc_target)

            source_ser.where(cond=source_ser > 0, other=0, inplace=True)

        except AttributeError as e:
            logger.info(f"AttributeError: {e}")

    def set_threshold(self, ent_name: str, threshold: float = 1e-10) -> None:
        """Set results to zero if value range is between zero an a given threshold."""
        ser = self.get(ent_name)
        setattr(self, ent_name, ser.where(cond=ser > threshold, other=0.0))
        self._changed_since_last_dic_export = True
