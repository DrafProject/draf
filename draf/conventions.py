"""Naming conventions for documentation and testing purpose

In general, entity names should adhere to the following structure:
    <Etype>_<Component>_<Descriptions>_<Dimensions>
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Alias:
    en: str
    de: str
    units: Optional[List] = None


# fmt: off
class Etypes:
    # SORTING_START
    A = Alias(en="Area", de="Fläche", units=["m²"])
    C = Alias(en="Costs", de="Kosten", units=["k€/a", "k€"])
    c = Alias(en="Specific costs", de="Spezifische Kosten", units=["€/kW", "€/kWh", "€/kW_th", "€/kWh_el", "€/kWh_th", "€/kW_el", "€/kW_el/a", "€/kW_peak", "€/tCO2eq", "€/SU", "€/t", "€/change"])
    CE = Alias(en="Carbon emissions", de="Kohlenstoff-vergleichs-Emissionen", units=["kgCO2eq/a"])
    ce = Alias(en="Specific carbon emissions", de="Spezifische Kohlenstoff-Emissionen", units=["kgCO2eq/kWh_el", "kgCO2eq/kWh"])
    cop = Alias(en="Coefficient of performance", de="Leistungszahl")
    dG = Alias(en="Product flow", de="Produkt fluss", units=["t/h"])
    dH = Alias(en="Hydrogen flow", de="Wasserstofffluss", units=["kW"])
    dQ = Alias(en="Heat flow", de="Wärmestrom", units=["kW_th", "kW"])
    E = Alias(en="Electrical energy", de="Elektrische Energie", units=["kWh_el"])
    eta = Alias(en="Efficiency", de="Effizienz", units=["", "kW_th/kW", "kW_el/kW", "kWh_th/kWh", "t/kWh_el"])
    F = Alias(en="Fuel", de="Brennstoff", units=["kW", "kWh"])
    G = Alias(en="Product", de="Produkt", units=["t"])
    H = Alias(en="Hydrogen", de="Wasserstoff", units=["kWh"])
    k = Alias(en="A ratio", de="Ein Verhältnis", units=["", "h", "m²/kW_peak"])
    n = Alias(en="A natural number", de="Eine natürliche Zahl")
    N = Alias(en="Operation life", de="Betriebsdauer", units=["a"])
    P = Alias(en="Electrical power", de="Elektrische Leistung", units=["kW_el", "kW_peak", "kW_el/kW_peak"])
    Q = Alias(en="Thermal Energy", de="Thermische Energie", units=["kWh_th"])
    T = Alias(en="Temperature", de="Temperatur", units=["°C"])
    t = Alias(en="Time", de="Zeit", units=["seconds", "h", "a"])
    X = Alias(en="A real number", de="Eine reelle Zahl")
    y = Alias(en="Binary indicator", de="Binärindikator")
    z = Alias(en="Binary allowance indicator", de="Binärindikator")
    # SORTING_END


class Descs:
    # SORTING_START
    amb = Alias(en="ambient", de="Umgebungs-")
    CAPn = Alias(en="New Capacity", de="Neue Kapazität")
    CAPx = Alias(en="Existing Capacity", de="Bestehende Kapazität")
    ch = Alias(en="Charging", de="Lade-")
    dis = Alias(en="Discharging", de="Entlade-")
    FI = Alias(en="Feed-in", de="Einspeisungsanteil")
    inv = Alias(en="Investment", de="Investitionen")
    invAnn = Alias(en="Annualized investment", de="Annualisierte Investitionen")
    MEF = Alias(en="Marginal Power Plant Emission Factors", de="Marginale CO2-Emissionsfaktoren des Stromsystems")
    OC = Alias(en="Own consumption", de="Eigenerzeugungsanteil")
    RMI = Alias(en="Repair, maintenance, and inspection per year and investment cost", de="Reparatur, Wartung und Inspektion pro Jahr und Investitionskosten")
    RTP = Alias(en="Real-time-prices", de="Dynamische Strompreise")
    TOU = Alias(en="Time-of-use", de="Zeitabhängige Strompreise")
    XEF = Alias(en="Average Electricity Mix Emission Factors", de="Durchschnittliche CO2-Emissionsfaktoren des Stromsystems")
    # SORTING_END


class Components:
    # SORTING_START
    BES = Alias(en="Battery energy storage", de="Batterie-Energiespeicher")
    BEV = Alias(en="Battery electric vehicle", de="Batterie-Elektrofahrzeug")
    cDem = Alias(en="Cooling demand", de="Kältebedarf")
    CHP = Alias(en="Combined heat and power", de="Kraft-Wärme-Kopplung (BHKW)")
    CM = Alias(en="Cooling machine", de="Kältemaschine")
    DAC = Alias(en="Direct air capture", de="CO2-Abscheidung")
    Dem = Alias(en="Demands", de="Bedarf")
    eDem = Alias(en="Electricity demand", de="Strombedarf")
    EG = Alias(en="Electricity grid", de="Stromnetz")
    Elc = Alias(en="Electrolyzer", de="Elektrolyseur")
    FC = Alias(en="Fuel cell", de="Brennstoffzelle")
    Fuel = Alias(en="Fuels", de="Brennstoffe")
    H2H = Alias(en="Heat downgrading", de="Wärmeabstufung")
    H2S = Alias(en="Hydrogen storage", de="Wasserstoffspeicher")
    hDem = Alias(en="Heat demand", de="Heizbedarf")
    HOB = Alias(en="Heat-only boiler", de="Heizkessel")
    HP = Alias(en="Heat pump", de="Wärmepumpe")
    HSB = Alias(en="High-speed steam boiler", de="Schnelldampferzeuger")
    P2H = Alias(en="Power to heat", de="Strom zu Wärme")
    PIT = Alias(en="Powered industrial truck", de="Flurförderfahrzeug")
    PP = Alias(en="Production process", de="Produktionsprozess")
    PS = Alias(en="Product storage", de="Produktlager")
    PV = Alias(en="Photovoltaic system", de="Fotovoltaik anlage")
    SB = Alias(en="Steam boiler", de="Dampfkessel")
    SN = Alias(en="Steam network", de="Dampfnetz")
    TES = Alias(en="Thermal energy storage", de="Thermischer speicher")
    TOT = Alias(en="Total", de="Total")
    WH = Alias(en="Waste heat", de="Abfallwärme")
    WT = Alias(en="Wind turbine", de="Windkraftanlage")
    # SORTING_END

# fmt: on
