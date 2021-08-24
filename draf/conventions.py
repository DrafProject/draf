from dataclasses import dataclass


@dataclass
class Acro:
    en: str
    de: str


class Types:
    A = Acro(en="Area", de="Fläche")
    C = Acro(en="Costs", de="Kosten")
    c = Acro(en="Specific costs", de="Spezifische Kosten")
    CE = Acro(en="Carbon emissions", de="Kohlenstoff-vergleichs-Emissionen")
    ce = Acro(en="Specific carbon emissions", de="Spezifische Kohlenstoff-Emissionen")
    cop = Acro(en="Coefficient of performance", de="Leistungszahl")
    E = Acro(en="Electrical energy", de="Elektrische Energie")
    eta = Acro(en="Efficiency", de="Effizienz")
    F = Acro(en="Fuel", de="Brennstoff")
    H = Acro(en="Heat flow", de="Wärmestrom")
    k = Acro(en="a ratio", de="ein Verhältnis")
    ol = Acro(en="Operation life", de="Betriebsdauer")
    P = Acro(en="Electrical power", de="Elektrische Leistung")
    Q = Acro(en="Thermal Energy", de="Thermische Energie")
    T = Acro(en="Temperature", de="Temperatur")
    y = Acro(en="Binary indicator", de="Binärindikator")
    z = Acro(en="Binary allowance indicator", de="Binärindikator")


class Acronyms:
    CAPn = Acro(en="New Capacity", de="Neue Kapazität")
    CAPx = Acro(en="Existing Capacity", de="Bestehende Kapazität")
    RMI = Acro(
        en="Repair, maintenance, and inspection per year and investment cost",
        de="Reparatur, Wartung und Inspektion pro Jahr und Investitionskosten",
    )
    RTP = Acro(en="Real-time-prices", de="Dynamische Strompreise")
    TOU = Acro(en="Time-of-use", de="Zeitabhängige Strompreise")
    FI = Acro(en="Feed-in", de="Einspeisungsanteil")
    OC = Acro(en="Own consumption", de="Eigenerzeugungsanteil")
    XEF = Acro(
        en="Average Electricity Mix Emission Factors",
        de="Durchschnittliche CO2-Emissionsfaktoren des Stromsystems",
    )
    MEF = Acro(
        en="Marginal Power Plant Emission Factors",
        de="Marginale CO2-Emissionsfaktoren des Stromsystems",
    )


class Components:
    BES = Acro(en="Battery energy storage", de="Batterie-Energiespeicher")
    BEV = Acro(en="Battery electric vehicle", de="Batterie-Elektrofahrzeug")
    CHP = Acro(en="Combined heat and power", de="Kraft-Wärme-Kopplung (BHKW)")
    CM = Acro(en="Cooling machine", de="Kältemaschine")
    CS = Acro(en="Cold storage", de="Kältespeicher")
    Dem = Acro(en="Demands", de="Bedarf")
    GRID = Acro(en="Electricity grid", de="Stromnetz")
    H2H = Acro(en="Heat downgrading", de="Wärmeabstufung")
    HOB = Acro(en="Heat only boiler", de="Heizkessel")
    HP = Acro(en="Heat pump", de="Wärmepumpe")
    HS = Acro(en="Heat storage", de="Wärmespeicher")
    HSB = Acro(en="High-Speed steam boiler", de="Schnelldampferzeuger")
    P2H = Acro(en="Power to heat", de="Strom zu Wärme")
    PIT = Acro(en="Powered industrial truck", de="Flurförderfahrzeug")
    PV = Acro(en="Photovoltaic", de="Fotovoltaik")
    SB = Acro(en="Steam boiler", de="Dampfkessel")
    SN = Acro(en="Steam network", de="Dampfnetz")
    WH = Acro(en="Waste heat", de="Abfallwärme")
