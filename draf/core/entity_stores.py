import logging
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from tabulate import tabulate

from draf import helper as hp
from draf.core.draf_base_class import DrafBaseClass

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)


def make_table(l: List[Tuple], lead_text: str = "", table_prefix="  "):
    headers, col_data = zip(*l)
    rows = list(zip(*col_data))
    return lead_text + textwrap.indent(
        text=tabulate(rows, headers=headers, floatfmt=".3n"), prefix=table_prefix
    )


class Collectors(DrafBaseClass):
    """Stores collectors."""

    def __init__(self):
        self._meta: Dict[str, Dict] = dict()

    def __repr__(self):
        data = self.get_all()
        meta = self._meta
        l = [
            ("Name", list(data.keys())),
            ("N", list(map(len, data.values()))),
            ("Content", [textwrap.shorten(", ".join(v.keys()), 50) for v in data.values()]),
            ("Unit", [meta[name]["unit"] for name in data.keys()]),
            ("Doc", [meta[name]["doc"] for name in data.keys()]),
        ]
        return make_table(l, lead_text="<Collectors object> preview:\n")


class Scenarios(DrafBaseClass):
    """Stores scenarios."""

    def __repr__(self):
        all = self.get_all()
        l = [
            ("Id", list(all.keys())),
            ("Name", [sc.name for sc in all.values()]),
            ("Doc", [sc.doc for sc in all.values()]),
        ]
        return make_table(l, lead_text="<Scenarios object> preview:\n")

    def rename(self, old_scen_id: str, new_scen_id: str) -> None:
        sc = self.__dict__.pop(old_scen_id)
        sc.id = new_scen_id
        self.__dict__[new_scen_id] = sc

    def get(self, scen_id) -> "Scenario":
        return getattr(self, scen_id)

    def get_by_name(self, name: str) -> "Scenario":
        for sc in self.get_all().values():
            if sc.name == name:
                return sc
        else:
            return None


class Dimensions(DrafBaseClass):
    """Stores dimensions."""

    def __init__(self):
        self._meta: Dict[str, Dict] = dict()

    def __repr__(self):
        data = self.get_all()
        meta = self._meta
        l = [
            ("Name", list(data.keys())),
            ("Doc", [meta[name]["doc"] for name in data.keys()]),
            ("Unit", [meta[name]["unit"] for name in data.keys()]),
        ]
        return make_table(l, lead_text="<Dimensions object> preview:\n")


class EntityStore(DrafBaseClass):
    def __repr__(self):
        layout = "{bullet}{name:<20} {etype:<4} {comp:<5} {dims:<5} {unit:<14} {doc}\n"
        return self._build_repr(layout, which_metadata=["unit", "etype", "comp", "doc", "dims"])

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
        etype: Optional[str] = None,
        comp: Optional[str] = None,
        desc: Optional[str] = None,
        dims: Optional[str] = None,
        func: Optional[Callable] = None,
    ) -> Dict:
        if func is None:
            func = lambda n: True
        return {
            k: v
            for k, v in self.get_all().items()
            if (
                (hp.get_etype(k) == etype or etype is None)
                and (hp.get_component(k) == comp or comp is None)
                and (hp.get_desc(k) == desc or desc is None)
                and (hp.get_dims(k) == dims or dims is None)
                and func(k)
            )
        }

    def get(self, name: str):
        """Returns entity"""
        return getattr(self, name)


class Params(EntityStore):
    """Stores parameters in a convenient way together with its functions."""

    def __init__(self):
        super().__init__()
        self._meta: Dict[str, Dict] = {}

    def __repr__(self):
        data = self.get_all()
        meta = self._meta
        l = [
            ("Name", list(data.keys())),
            ("Dims", [hp.get_dims(k) for k in data.keys()]),
            ("(⌀) Value", [hp.get_mean(i) for i in data.values()]),
            ("Unit", [meta[k]["unit"] for k in data.keys()]),
            ("Doc", [textwrap.fill(meta[k]["doc"], width=40) for k in data.keys()]),
            ("Source", [textwrap.shorten(meta[k]["src"], width=17) for k in data.keys()]),
        ]
        return make_table(l, lead_text=f"<Params object> preview:\n")

    def _set_meta(self, ent_name: str, meta_type: str, value: str) -> None:
        self._meta.setdefault(ent_name, {})[meta_type] = value

    def _convert_unit(
        self,
        ent_name: str,
        return_unit: str,
        conversion_factor: float = None,
        conversion_func: Callable = None,
    ):
        par = self.get(ent_name)
        if conversion_factor is not None:
            par *= conversion_factor
        if conversion_func is not None:
            par = conversion_func(par)
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

    def __repr__(self):
        data = self.get_all()
        meta = self._meta
        l = [
            ("Name", list(data.keys())),
            ("Dims", [hp.get_dims(k) for k in data.keys()]),
            ("(⌀) Value", [hp.get_mean(i) for i in data.values()]),
            ("Unit", [meta[k]["unit"] for k in data.keys()]),
            ("Doc", [textwrap.fill(meta[k]["doc"], width=50) for k in data.keys()]),
        ]
        return make_table(l, lead_text="<Results object> preview:\n")

    # TODO: Move _get_results_from_variables, _from_gurobipy, _from_pyomo to Scenario for
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

            if target_neg is None:
                if source_ser.min() < -0.1:
                    logger.warning(
                        f"Significant negative values (between {source_ser.min():n} and 0) of the"
                        f" entity '{source}' were clipped"
                    )

            else:
                ser = -source_ser.where(cond=source_ser < 0, other=0)
                ser.name = target_neg
                setattr(self, target_neg, ser)

                which_metas = ["etype", "comp", "unit", "dims"]
                self._copy_meta(source_ent=source, target_ent=target_neg, which_metas=which_metas)

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
