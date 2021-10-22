# This file was copied from the gsee repository
# https://github.com/renewables-ninja/gsee/tree/2738600e64a645f97eb96f6c9fb7d3f2856cf24c
# as a workaround to https://github.com/renewables-ninja/gsee/issues/12

# This is the license of GSEE:

# BSD 3-Clause License

# Copyright (c) 2013-2018, Stefan Pfenninger
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pvlib


def get_efficiency(irradiance, cell_temperature, module_params):
    """
    irradiance : float or pandas.Series
        Effective irradiance (W/m2) that is converted to photocurrent.
    cell_temperature : float or pandas.Series
        Average cell temperature of cells within a module in deg C.
    module_params : dict
        Module params 'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s'.

    """
    params = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance=irradiance, temp_cell=cell_temperature, **module_params
    )

    # Ensure that the shunt resistance is not infinite
    # Commented out because we want to still return valid Series when
    # some of the values are zero -- NaNs from 0-divisions are filled later
    # assert params[3] != math.inf

    dc = pvlib.pvsystem.singlediode(*params)
    efficiency = dc["p_mp"] / irradiance
    return efficiency


def relative_eff(irradiance, cell_temperature, params):
    """
    Compute relative efficiency of PV module as a function of irradiance
    and cell/module temperature, from Huld (2010):

    .. math:: n_{rel} = \frac{P_{stc} * (G / G_{stc})}{P}

    Where G is in-plane irradiance, P is power output,
    and STC conditions are :math:`G = 1000` and
    :math:`T_{mod} = 25`.

    When irradiance is zero, a zero relative efficiency is returned.

    Parameters
    ----------

    irradiance : float or pandas.Series
        Irradiance in W/m2.
    cell_temperature : float or pandas.Series
        Average cell temperature of cells within a module in deg C.
    params : dict
        Module params 'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s'.

    """
    if isinstance(irradiance, float) and irradiance == 0:
        return 0

    power_stc = 1000 * get_efficiency(1000, 25, params)
    power = irradiance * get_efficiency(irradiance, cell_temperature, params)

    # Fill NaNs from any possible divisions by zero with 0
    return (power / (power_stc * (irradiance / 1000))).fillna(0)
