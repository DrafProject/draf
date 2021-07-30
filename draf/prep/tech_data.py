from dataclasses import dataclass
from typing import List, Union

import pandas as pd


@dataclass
class Entry:
    name: str
    data: Union[str, List]
    doc: str = ""
    src: str = ""
    unit: str = ""

    @property
    def type(self) -> str:
        return self.name.split("_")[0]

    @property
    def component(self) -> str:
        return self.name.split("_")[1]

    @property
    def dims(self) -> str:
        return self.name.split("_")[2]


class SRC:
    BMWI_2020 = (
        "https://www.bmwi-energiewende.de/EWD/Redaktion/Newsletter/2020/11/Meldung/News1.html"
    )
    BRACCO_2016 = "https://doi.org/10.1016/j.energy.2016.01.050"
    FFE_2016 = (
        "https://www.ffe.de/images/stories/Themen/414_MOS/20160728_MOS_Speichertechnologien.pdf"
    )
    FIGGENER_2020 = "https://doi.org/10.1016/j.est.2020.101982"
    GEG9_2020 = "https://www.buzer.de/Anlage_9_GEG.htm"
    IFEU_2008 = "https://www.ifeu.de/oekobilanzen/pdf/THG_Bilanzen_Bio_Erdgas.pdf"
    ISE_2018 = "https://www.ise.fraunhofer.de/content/dam/ise/de/documents/publications/studies/DE2018_ISE_Studie_Stromgestehungskosten_Erneuerbare_Energien.pdf"
    JUELCH_2016 = "https://doi.org/10.1016/j.apenergy.2016.08.165"
    KALTSCH_2020 = "https://doi.org/10.1007/978-3-662-61190-6"
    MTA = "MTA Galaxy tech GLT 210/SSN in https://www.mta.de/fileadmin/user_upload/MTA-Produktuebersicht-Prozesskuehlung.pdf"
    PLAN_BIOGAS = "https://planet-biogas.de/biogasprodukte/bhkw/"
    SMT_2018 = "https://www.strommarkttreffen.org/2018-02_Hinterberger_P2H-Anlagen_zur_Verwertung_von_EE-Ueberschussstrom.pdf"
    VDI2067 = "https://www.beuth.de/de/technische-regel/vdi-2067-blatt-1/151420393"


@dataclass
class Acro:
    en: str
    de: str


class Acronyms:
    BES = Acro(en="Battery energy storage", de="Batterie-Energiespeicher")
    BEV = Acro(en="Battery electric vehicle", de="Batterie-Elektrofahrzeug")
    C = Acro(en="Costs", de="Kosten")
    CAPn = Acro(en="New Capacity", de="Neue Kapazität")
    CAPx = Acro(en="Existing Capacity", de="Bestehende Kapazität")
    CE = Acro(en="Carbon emissions", de="CO2-vergleichs-Emissionen")
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
    RTP = Acro(en="Real-time-prices", de="Dynamische Strompreise")
    SB = Acro(en="Steam boiler", de="Dampfkessel")
    SN = Acro(en="Steam network", de="Dampfnetz")
    TOU = Acro(en="Time-of-use", de="Zeitabhängige Strompreise")
    WH = Acro(en="Waste heat", de="Abfallwärme")


RMI = "Repair, maintenance, and inspection per year per investment cost."


class TechData:
    c_EEG_ = Entry(name="c_EEG_", data=0.065, doc=f"EEG levy", src=SRC.BMWI_2020, unit="€/kWh_el")
    c_FUEL_co2_ = Entry(
        name="c_FUEL_co2_", data=55, doc="CO2 price for non-electricity", unit="€/tCO2eq"
    )
    c_FUEL_F = Entry(
        name="c_FUEL_F", data=pd.Series({"ng": 0.04, "bio": 0.02}), doc="Fuel cost", unit="€/kWh"
    )
    c_P2H_inv_ = Entry(
        name="c_P2H_inv_", data=100, doc=f"System CAPEX.", src=SRC.SMT_2018, unit="€/kW_th"
    )
    
    ce_FUEL_F = Entry(
        name="ce_FUEL_F",
        data=pd.Series({"ng": 0.240, "bio": 0.075}),
        doc=f"Fuel carbon emissions.",
        src=f"{SRC.GEG9_2020} documentation in {SRC.IFEU_2008}",
        unit="kgCO2eq/kWh",
    )
    cop_CM_ = Entry(
        name="cop_CM_",
        data=3.98,
        doc="Coefficient of performance.",
        src=SRC.MTA,
        unit="kWh_th/kW_el",
    )

    eta_CHP_el_ = Entry(
        name="eta_CHP_el_",
        data=0.42,
        doc=f"Electric efficiency.",
        src=SRC.PLAN_BIOGAS,
        unit="kW_el/kW",
    )
    eta_CHP_th_ = Entry(
        name="eta_CHP_th_",
        data=0.42,
        doc=f"Thermal efficiency.",
        src=SRC.PLAN_BIOGAS,
        unit="kW_th/kW",
    )
    eta_CS_in_ = Entry(name="eta_CS_in_", data=0.99, doc=f"Loading efficiency.", src=SRC.FFE_2016)
    eta_CS_time_ = Entry(
        name="eta_CS_time_", data=0.95, doc=f"Storing efficiency.", src=SRC.FFE_2016
    )
    eta_HP_ = Entry(
        name="eta_HP_", data=0.5, doc=f"Ratio of reaching the ideal COP.", src=SRC.KALTSCH_2020
    )
    eta_HS_in_ = Entry(name="eta_HS_in_", data=0.99, doc=f"Loading efficiency.", src=SRC.FFE_2016)
    eta_HS_time_ = Entry(
        name="eta_HS_time_", data=0.95, doc=f"Storing efficiency.", src=SRC.FFE_2016
    )
    k_BES_inPerCapa_ = Entry(
        name="k_BES_inPerCapa_",
        data=0.7,
        doc=f"Ratio charging power / capacity.",
        src=SRC.FIGGENER_2020,
    )

    k_BES_RMI_ = Entry(name="k_BES_RMI_", data=0.02, doc=RMI, src=SRC.JUELCH_2016)
    k_CHP_RMI_ = Entry(name="k_CHP_RMI_", data=0.06 + 0.02, doc=RMI, src=SRC.VDI2067)
    k_CM_RMI_ = Entry(name="k_CM_RMI_", data=0.01 + 0.015, doc=RMI, src=SRC.VDI2067)
    k_CS_RMI_ = Entry(name="k_CS_RMI_", data=0.001, doc=RMI, src=SRC.FFE_2016)
    k_HP_RMI_ = Entry(name="k_HP_RMI_", data=0.01 + 0.015, doc=RMI, src=SRC.VDI2067)
    k_HS_RMI_ = Entry(name="k_HS_RMI_", data=0.001, doc=RMI, src=SRC.FFE_2016)
    k_PV_RMI_ = Entry(name="k_PV_RMI_", data=0.02, doc=RMI, src=SRC.ISE_2018)

    ol_BES_ = Entry(name="ol_BES_", data=20, doc=f"Operation life.", src=SRC.JUELCH_2016, unit="a")
    ol_CHP_ = Entry(name="ol_CHP_", data=15, doc=f"Operation life.", src=SRC.VDI2067, unit="a")
    ol_CS_ = Entry(name="ol_CS_", data=30, doc=f"Operation life.", src=SRC.BRACCO_2016, unit="a")
    ol_HP_ = Entry(name="ol_HP_", data=18, doc=f"Operation life.", src=SRC.VDI2067, unit="a")
    ol_HS_ = Entry(name="ol_HS_", data=30, doc=f"Operation life.", src=SRC.BRACCO_2016, unit="a")
    ol_PV_ = Entry(name="ol_PV_", data=25, doc=f"Operation life.", src=SRC.ISE_2018, unit="a")
