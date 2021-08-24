import pandas as pd

from draf.prep.par_dat import ParDat


# fmt: off
class SRC:
    ASUE_2011 = "https://asue.de/sites/default/files/asue/themen/blockheizkraftwerke/2011/broschueren/05_07_11_asue-bhkw-kenndaten-0311.pdf, p.12"
    BMWI_2020 = "https://www.bmwi-energiewende.de/EWD/Redaktion/Newsletter/2020/11/Meldung/News1.html"
    BRACCO_2016 = "https://doi.org/10.1016/j.energy.2016.01.050"
    CARROQUINO_2021 = "https://doi.org/10.3390/app11083587"
    FFE_2016 = "https://www.ffe.de/images/stories/Themen/414_MOS/20160728_MOS_Speichertechnologien.pdf"
    FIGGENER_2020 = "https://doi.org/10.1016/j.est.2020.101982"
    GEG9_2020 = "https://www.buzer.de/Anlage_9_GEG.htm"
    IFEU_2008 = "https://www.ifeu.de/oekobilanzen/pdf/THG_Bilanzen_Bio_Erdgas.pdf"
    ISE_2018 = "https://www.ise.fraunhofer.de/content/dam/ise/de/documents/publications/studies/DE2018_ISE_Studie_Stromgestehungskosten_Erneuerbare_Energien.pdf"
    JUELCH_2016 = "https://doi.org/10.1016/j.apenergy.2016.08.165"
    KALTSCH_2020 = "https://doi.org/10.1007/978-3-662-61190-6"
    MATHIESEN_2015 = "https://doi.org/10.1016/j.apenergy.2015.01.075"
    MTA = "MTA Galaxy tech GLT 210/SSN in https://www.mta.de/fileadmin/user_upload/MTA-Produktuebersicht-Prozesskuehlung.pdf"
    PLAN_BIOGAS = "https://planet-biogas.de/biogasprodukte/bhkw/"
    PVMAG_2020 = "https://www.pv-magazine.de/2020/03/13/pv-magazine-marktuebersicht-fuer-grossspeicher-aktualisiert"
    REDONDO_2016 = "https://doi.org/10.1109%2FVPPC.2016.7791723"
    SMT_2018 = "https://www.strommarkttreffen.org/2018-02_Hinterberger_P2H-Anlagen_zur_Verwertung_von_EE-Ueberschussstrom.pdf"
    VARTIAINEN_2019 = "https://doi.org/10.1002/pip.3189"
    VDI2067 = "https://www.beuth.de/de/technische-regel/vdi-2067-blatt-1/151420393"
    VIESSMANN = "http://www.heizungs-discount.de/Kataloge/Viessmann/viessmann_preisliste_mittel_und_grosskessel_vitoplex_vitocell_vitorond_vitotrans_vitoradial_vitomax_vitoplex_vitocontrol_kwt_vitocal_pyrot_pyrotec_mawera_vitocom_vitodata_vitohome.pdf"
    WOLF_2017 = "https://doi.org/10.18419/opus-9593"
# fmt: on


class DataBase:
    from draf.conventions import Acronyms
    from draf.prep import param_funcs as funcs

    c_EEG_ = ParDat(name="c_EEG_", data=0.065, doc=f"EEG levy", src=SRC.BMWI_2020, unit="€/kWh_el")
    c_FUEL_co2_ = ParDat(
        name="c_FUEL_co2_", data=55, doc="CO2 price for non-electricity", unit="€/tCO2eq"
    )
    c_FUEL_F = ParDat(
        name="c_FUEL_F", data=pd.Series({"ng": 0.04, "bio": 0.02}), doc="Fuel cost", unit="€/kWh"
    )
    c_P2H_inv_ = ParDat(
        name="c_P2H_inv_", data=100, doc=f"System CAPEX.", src=SRC.SMT_2018, unit="€/kW_th"
    )

    ce_FUEL_F = ParDat(
        name="ce_FUEL_F",
        data=pd.Series({"ng": 0.240, "bio": 0.075}),
        doc=f"Fuel carbon emissions.",
        src=f"{SRC.GEG9_2020} documentation in {SRC.IFEU_2008}",
        unit="kgCO2eq/kWh",
    )
    cop_CM_ = ParDat(
        name="cop_CM_",
        data=3.98,
        doc="Coefficient of performance.",
        src=SRC.MTA,
        unit="kWh_th/kW_el",
    )

    eta_CS_in_ = ParDat(name="eta_CS_in_", data=0.99, doc=f"Loading efficiency.", src=SRC.FFE_2016)
    eta_CS_time_ = ParDat(
        name="eta_CS_time_", data=0.95, doc=f"Storing efficiency.", src=SRC.FFE_2016
    )
    eta_HP_ = ParDat(
        name="eta_HP_", data=0.5, doc=f"Ratio of reaching the ideal COP.", src=SRC.KALTSCH_2020
    )
    eta_HS_in_ = ParDat(name="eta_HS_in_", data=0.99, doc=f"Loading efficiency.", src=SRC.FFE_2016)
    eta_HS_time_ = ParDat(
        name="eta_HS_time_", data=0.95, doc=f"Storing efficiency.", src=SRC.FFE_2016
    )
    k_BES_inPerCapa_ = ParDat(
        name="k_BES_inPerCapa_",
        data=0.7,
        doc=f"Ratio charging power / capacity.",
        src=SRC.FIGGENER_2020,
    )
    eta_BES_time_ = ParDat(
        name="eta_BES_time_",
        # "0.35% to 2.5% per month depending on state of charge"
        data=1 - (0.0035 + 0.024) / 2 / (30 * 24),
        doc="Efficiency due to self-discharge rate",
        src=SRC.REDONDO_2016,
    )
    eta_BES_in_ = ParDat(
        name="eta_BES_in_", data=0.95, doc="Cycling efficiency", src=SRC.CARROQUINO_2021
    )

    k_BES_RMI_ = ParDat(name="k_BES_RMI_", data=0.02, doc=Acronyms.RMI.en, src=SRC.JUELCH_2016)
    k_CHP_RMI_ = ParDat(name="k_CHP_RMI_", data=0.06 + 0.02, doc=Acronyms.RMI.en, src=SRC.VDI2067)
    k_CM_RMI_ = ParDat(name="k_CM_RMI_", data=0.01 + 0.015, doc=Acronyms.RMI.en, src=SRC.VDI2067)
    k_CS_RMI_ = ParDat(name="k_CS_RMI_", data=0.001, doc=Acronyms.RMI.en, src=SRC.FFE_2016)
    k_HOB_RMI_ = ParDat(name="k_HOB_RMI_", data=0.04, doc=Acronyms.RMI.en)  # TODO
    k_HP_RMI_ = ParDat(name="k_HP_RMI_", data=0.01 + 0.015, doc=Acronyms.RMI.en, src=SRC.VDI2067)
    k_HS_RMI_ = ParDat(name="k_HS_RMI_", data=0.001, doc=Acronyms.RMI.en, src=SRC.FFE_2016)
    k_PV_RMI_ = ParDat(name="k_PV_RMI_", data=0.02, doc=Acronyms.RMI.en, src=SRC.ISE_2018)

    ol_BES_ = ParDat(name="ol_BES_", data=20, doc=f"Operation life.", src=SRC.JUELCH_2016, unit="a")
    ol_CHP_ = ParDat(name="ol_CHP_", data=15, doc=f"Operation life.", src=SRC.VDI2067, unit="a")
    ol_CS_ = ParDat(name="ol_CS_", data=30, doc=f"Operation life.", src=SRC.BRACCO_2016, unit="a")
    ol_HP_ = ParDat(name="ol_HP_", data=18, doc=f"Operation life.", src=SRC.VDI2067, unit="a")
    ol_HS_ = ParDat(name="ol_HS_", data=30, doc=f"Operation life.", src=SRC.BRACCO_2016, unit="a")
    ol_PV_ = ParDat(name="ol_PV_", data=25, doc=f"Operation life.", src=SRC.ISE_2018, unit="a")
