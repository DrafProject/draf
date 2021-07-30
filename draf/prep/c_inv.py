"""Functions that provide investment prices for energy-related technologies in the industrial
environment.
"""
import pandas as pd

from draf.prep.data_base import ParDat


def c_PV_inv_() -> ParDat:
    """CAPEX for PV

    Valid for Europe in 2019 for P_inst=>500 kW_peak.
    """
    src = "Vartiainen et al (2019): https://doi.org/10.1002/pip.3189"
    return ParDat(name="c_PV_inv_", data=460, doc="CAPEX", src=src, unit="€/kW_peak")


def c_BES_inv_(estimated_size=100, which="mean") -> ParDat:
    """CAPEX for lithium-ion battery energy storages

    Parameters:
        estimated_size: 0..50000 kWh_el installed capacity.
        which: Selects in ('lowest', 'mean', 'highest') of the realized prices.
    """
    src = "pv magazine(2020): https://www.pv-magazine.de/2020/03/13/pv-magazine-marktuebersicht-fuer-grossspeicher-aktualisiert/"
    assert 0 <= estimated_size <= 50000
    assert which in ("lowest", "mean", "highest")

    sizes = (100, 500, 1000, 10000, 50000)
    prices = dict(
        highest=(990, 980, 700, 650, 600),
        mean=(720, 651, 550, 497, 413),
        lowest=(295, 280, 250, 245, 215),
    )

    for size, price in zip(sizes, prices[which]):
        if estimated_size <= size:
            break
    return ParDat(name="c_BES_inv_", data=price, doc="CAPEX", src=src, unit="€/kWh_el")


def c_CHP_inv_(estimated_size=400, fuel_type="bio") -> ParDat:
    """CAPEX for combined heat and power

    Parameters:
        estimated_size: 0..2500 kW_el nominal electric power.
        fuel_type: In['bio', 'ng']

    Formulas:
        ng: 9332.6 * estimated_size ** -0.461
        bio: 15648 * estimated_size ** -0.5361
    """
    src = "ASUE(2011): https://asue.de/sites/default/files/asue/themen/blockheizkraftwerke/2011/broschueren/05_07_11_asue-bhkw-kenndaten-0311.pdf, p.12"
    assert 0 <= estimated_size <= 2500

    if fuel_type == "ng":
        value = 9332.6 * estimated_size ** -0.461
    elif fuel_type == "bio":
        value = 15648 * estimated_size ** -0.5361
    else:
        raise ValueError("fuel_type must be in ['bio', 'ng'].")

    return ParDat(name="c_CHP_inv_", data=value, doc="CAPEX", src=src, unit="€/kW_el")


def c_HP_inv_(estimated_size=100) -> ParDat:
    """CAPEX for electric heat pumps

    Parameters:
        estimated_size: 0..200 kWh_th heating power.

    Formula:
        1520.7 * estimated_size ** -.363
    """
    src = "Wolf(2017): http://dx.doi.org/10.18419/opus-9593"
    assert 0 < estimated_size < 200
    value = 1520.7 * estimated_size ** -0.363
    return ParDat(name="c_HP_inv_", data=value, doc="CAPEX", src=src, unit="€/kW_el")


def c_HS_inv_(estimated_size=100, temp_spread=40) -> ParDat:
    """CAPEX for heat storages.

    Parameters:
        estimated_size: 30..30000 m³ storage size.
        temp_spread: Temperature spread in °C.

    Formula:
        price_per_m3 = 8222.6 * estimated_size ** -0.394
    """
    src = "FFE(2016): https://www.ffe.de/images/stories/Themen/414_MOS/20160728_MOS_Speichertechnologien.pdf, page 68"
    assert 30 < estimated_size < 30000
    assert 0 < temp_spread < 1000
    specific_heat = 4.2  # kJ/(kg*K)
    kJ_per_kWh = 3600  # kJ/kWh
    density = 999.975  # kg/m³
    kWh_per_m3 = specific_heat / kJ_per_kWh * temp_spread * density
    price_per_m3 = 8222.6 * estimated_size ** -0.394
    value = price_per_m3 / kWh_per_m3
    return ParDat(name="c_HS_inv_", data=value, doc="CAPEX", src=src, unit="€/kW_th")


def c_HOB_inv_() -> ParDat:
    """"CAPEX for gas boiler Vitoplex 300 with 620 kWh_th"""
    src = "http://www.heizungs-discount.de/Kataloge/Viessmann/viessmann_preisliste_mittel_und_grosskessel_vitoplex_vitocell_vitorond_vitotrans_vitoradial_vitomax_vitoplex_vitocontrol_kwt_vitocal_pyrot_pyrotec_mawera_vitocom_vitodata_vitohome.pdf"
    thermal_capa_in_kW_th = 620
    cost_factor_for_pipes_etc = 1.5
    main_component_costs = pd.Series({"Heizkessel": 18216, "Weishaupt Gebläsebrenner": 5399})
    value = cost_factor_for_pipes_etc * main_component_costs.sum() / thermal_capa_in_kW_th
    return ParDat(name="c_HS_inv_", data=value, doc="CAPEX", src=src, unit="€/kW_th")
