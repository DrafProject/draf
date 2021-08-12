from typing import List, Tuple

from draf import Params, Vars
from draf import helper as hp


def _agg_cap(capEntName: str, v: Vars) -> float:
    """Returns the aggregated new capacity of an entity such as `P_HP_CAPn_N` or `E_BES_CAPn_`."""
    if hp.get_dims(capEntName) == "":
        return v.get(capEntName)
    else:
        return v.get(capEntName).sum()


def _capas(v: Vars) -> List[Tuple[str, str]]:
    """Returns the new capacities per component type"""
    return {hp.get_component(key): _agg_cap(key, v) for key in v.filtered(acro="CAPn")}


def C_inv_(p: Params, v: Vars):
    """Returns the sum product of all scalar capacities and investment costs.
    WARNING: The model must contain scalar investment prices `c_<COMPONENT>_inv_`.

    Example:
        >>> model.addConstr((v.C_inv_ == collectors.C_inv_(p, v)))
    """
    return sum([cap * p.get(f"c_{c}_inv_") for c, cap in _capas(v).items()])


def C_invAnnual_(p: Params, v: Vars):
    """Returns the annualized investment costs.
    WARNING: The model must contain scalar investment prices `c_<COMPONENT>_inv_`.

    Example:
        >>> model.addConstr((v.C_invAnnual_ == collectors.C_invAnnual_(p, v)))

    """
    return sum([cap * p.get(f"c_{c}_inv_") / p.get(f"ol_{c}_") for c, cap in _capas(v).items()])


def C_RMI_(p: Params, v: Vars):
    """Returns a linear expression for the repair, maintenance, and inspection per year.
    WARNING: The model must contain scalar parameters for investment prices `c_<COMPONENT>_inv_`.

    Example:
        >>> model.addConstr((v.C_RMI_ == collectors.C_RMI_(p, v)))
    """
    return sum([cap * p.get(f"c_{c}_inv_") * p.get(f"k_{c}_RMI_") for c, cap in _capas(v).items()])
