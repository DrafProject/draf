from dataclasses import dataclass


@dataclass
class Alias:
    en: str
    de: str


# fmt: off
class Etypes:
    # SORTING_START
    A = Alias(en="Area", de="Fläche")
    C = Alias(en="Costs", de="Kosten")
    CE = Alias(en="Carbon emissions", de="Kohlenstoff-vergleichs-Emissionen")
    E = Alias(en="Electrical energy", de="Elektrische Energie")
    F = Alias(en="Fuel", de="Brennstoff")
    P = Alias(en="Electrical power", de="Elektrische Leistung")
    Q = Alias(en="Thermal Energy", de="Thermische Energie")
    T = Alias(en="Temperature", de="Temperatur")
    c = Alias(en="Specific costs", de="Spezifische Kosten")
    ce = Alias(en="Specific carbon emissions", de="Spezifische Kohlenstoff-Emissionen")
    cop = Alias(en="Coefficient of performance", de="Leistungszahl")
    dQ = Alias(en="Heat flow", de="Wärmestrom")
    eta = Alias(en="Efficiency", de="Effizienz")
    k = Alias(en="a ratio", de="ein Verhältnis")
    ol = Alias(en="Operation life", de="Betriebsdauer")
    y = Alias(en="Binary indicator", de="Binärindikator")
    z = Alias(en="Binary allowance indicator", de="Binärindikator")
    # SORTING_END


class Descs:
    # SORTING_START
    CAPn = Alias(en="New Capacity", de="Neue Kapazität")
    CAPx = Alias(en="Existing Capacity", de="Bestehende Kapazität")
    FI = Alias(en="Feed-in", de="Einspeisungsanteil")
    MEF = Alias(en="Marginal Power Plant Emission Factors", de="Marginale CO2-Emissionsfaktoren des Stromsystems")
    OC = Alias(en="Own consumption", de="Eigenerzeugungsanteil")
    RMI = Alias(en="Repair, maintenance, and inspection per year and investment cost", de="Reparatur, Wartung und Inspektion pro Jahr und Investitionskosten")
    RTP = Alias(en="Real-time-prices", de="Dynamische Strompreise")
    TOU = Alias(en="Time-of-use", de="Zeitabhängige Strompreise")
    XEF = Alias(en="Average Electricity Mix Emission Factors", de="Durchschnittliche CO2-Emissionsfaktoren des Stromsystems")
    # SORTING_END


class Components:
    # SORTING_START
    BES = Alias(en="Battery energy storage", de="Batterie-Energiespeicher")
    BEV = Alias(en="Battery electric vehicle", de="Batterie-Elektrofahrzeug")
    CHP = Alias(en="Combined heat and power", de="Kraft-Wärme-Kopplung (BHKW)")
    CM = Alias(en="Cooling machine", de="Kältemaschine")
    Dem = Alias(en="Demands", de="Bedarf")
    EG = Alias(en="Electricity grid", de="Stromnetz")
    H2H = Alias(en="Heat downgrading", de="Wärmeabstufung")
    HOB = Alias(en="Heat only boiler", de="Heizkessel")
    HP = Alias(en="Heat pump", de="Wärmepumpe")
    HSB = Alias(en="High-Speed steam boiler", de="Schnelldampferzeuger")
    P2H = Alias(en="Power to heat", de="Strom zu Wärme")
    PIT = Alias(en="Powered industrial truck", de="Flurförderfahrzeug")
    PV = Alias(en="Photovoltaic", de="Fotovoltaik")
    SB = Alias(en="Steam boiler", de="Dampfkessel")
    SN = Alias(en="Steam network", de="Dampfnetz")
    TES = Alias(en="Thermal energy storage", de="Thermischer speicher")
    WH = Alias(en="Waste heat", de="Abfallwärme")
    # SORTING_END

# fmt: on
