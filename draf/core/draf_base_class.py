from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import textwrap


class DrafBaseClass:
    """This class mainly provides functions."""

    @staticmethod
    def _get_dims(ent: str) -> str:
        """Returns the dimensions of a given entity name, based on its name."""
        lst = ent.split("_")
        assert len(
            lst) > 0, "Every entity must contain at least one underscore which '{ent}' has not."
        return lst[-1]

    def _build_repr(self, layout: str = None, which_metadata: Optional[List] = None) -> str:

        preface = "<{} object>".format(self.__class__.__name__)

        self_mapping = {k: k.capitalize() for k in which_metadata}
        header = layout.format(bullet="    ", name="Name", **self_mapping)
        header += f"{'='*79}\n"
        this_list = []

        if hasattr(self, "_meta"):
            for ent_name, meta_obj in self._meta.items():
                metas = {meta_type: meta_obj.get(meta_type, None) for meta_type in which_metadata}
                try:
                    metas["doc"] = textwrap.shorten(metas["doc"], width=68)
                except KeyError:
                    pass
                this_list.append(layout.format(bullet="  ⤷ ", name=ent_name, **metas))

        else:
            for k, v in self.get_all().items():
                metas = {meta_type: getattr(v, meta_type) for meta_type in which_metadata}
                this_list.append(layout.format(bullet="  ⤷ ", name=k, **metas))

        if this_list:
            data = "".join(this_list)
            return f"{preface}\n{header}{data}"
        else:
            return f"{preface} (empty)"

    def get_all(self) -> Dict:
        """Returns a Dict with all public non-callable attributes from this container."""
        return {k: v for k, v in vars(self).items() if not (k.startswith('_') or callable(v))}

    def get(self, name: str):
        """Returns entity"""
        return getattr(self, name)
