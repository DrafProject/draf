from typing import Dict, List, Tuple

from draf import Params, Vars
from draf import helper as hp


def _agg_cap(capEntName: str, v: Vars) -> float:
    """Return aggregated new capacity of an entity such as `P_HP_CAPn_N` or `E_BES_CAPn_`."""
    if hp.get_dims(capEntName) == "":
        return v.get(capEntName)
    else:
        return v.get(capEntName).sum()


def _capas(v: Vars) -> Dict[str, float]:
    """Return new capacities per component type."""
    return {hp.get_component(key): _agg_cap(key, v) for key in v.filtered(desc="CAPn")}


def C_TOT_inv_(p: Params, v: Vars):
    """Return sum product of all scalar capacities and investment costs.

    Example:
        >>> model.addConstr((v.C_TOT_inv_ == collectors.C_TOT_inv_(p, v)))
    """
    return sum([cap * p.get(f"c_{c}_inv_") for c, cap in _capas(v).items()])


def C_invAnnual_(p: Params, v: Vars, r: float):
    """Return annualized investment costs.

    Example:
        >>> model.addConstr((v.C_invAnnual_ == collectors.C_invAnnual_(p, v)))

    """
    return sum(
        [
            cap * p.get(f"c_{c}_inv_") * hp.get_annuity_factor(r=r, N=p.get(f"N_{c}_"))
            for c, cap in _capas(v).items()
        ]
    )


def C_TOT_RMI_(p: Params, v: Vars):
    """Return linear expression for the repair, maintenance, and inspection per year.

    Example:
        >>> model.addConstr((v.C_TOT_RMI_ == collectors.C_TOT_RMI_(p, v)))
    """
    return sum([cap * p.get(f"c_{c}_inv_") * p.get(f"k_{c}_RMI_") for c, cap in _capas(v).items()])
