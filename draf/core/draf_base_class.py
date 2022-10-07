import textwrap
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class DrafBaseClass:
    """This class mainly provides functions."""

    def _build_repr(self, layout: str, which_metadata: Iterable) -> str:

        preface = f"<{self.__class__.__name__} object>"

        self_mapping = {k: k.capitalize() for k in which_metadata}
        header = layout.format(bullet="    ", name="Name", **self_mapping)
        header += f"{'='*100}\n"
        this_list = []

        if hasattr(self, "_meta"):
            for ent_name, meta_obj in self._meta.items():
                metas = {meta_type: meta_obj.get(meta_type, None) for meta_type in which_metadata}
                try:
                    metas["doc"] = textwrap.shorten(metas["doc"], width=68)
                except KeyError:
                    pass
                try:
                    appender = layout.format(bullet="  ⤷ ", name=ent_name, **metas)
                except TypeError:
                    appender = f"  ⤷ {ent_name} ====No Metadata found===="
                this_list.append(appender)

        else:
            for k, v in self.get_all().items():
                metas = {meta_type: getattr(v, meta_type) for meta_type in which_metadata}
                this_list.append(layout.format(bullet="  ⤷ ", name=k, **metas))

        if this_list:
            data = "".join(this_list)
            return f"{preface} preview:\n{header}{data}"
        else:
            return f"{preface} (empty)"

    def get_all(self) -> Dict:
        """Returns a Dict with all public attributes from this container."""
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_"))}

    def __iter__(self):
        return iter(self.get_all().values())

    def __len__(self):
        return len(self.get_all())
