import pandas as pd

from draf.prep.par_dat import ParDat

# fmt: off

# This file is ignored from the black formatter.
# Data entries are alphabetically sorted within the formatting routine.

class SRC:
    # SORTING_START
    ASUE_2011 = "https://perma.cc/KHG2-WPPX, p.12"
    BMWI_2020 = "https://www.bmwi-energiewende.de/EWD/Redaktion/Newsletter/2020/11/Meldung/News1.html"
    BRACCO_2016 = "https://doi.org/10.1016/j.energy.2016.01.050"
    Carroquino_2021 = "https://doi.org/10.3390/app11083587"
    FFE_2016 = "https://perma.cc/F9X5-HG3B"
    Figgener_2021 = "https://doi.org/10.1016/j.est.2020.101982"
    GEG9_2020 = "https://www.buzer.de/Anlage_9_GEG.htm"
    IFEU_2008 = "https://perma.cc/UZX6-ZN73"
    ISE_2018 = "https://perma.cc/MQS2-DCPL"
    Juelch_2016 = "https://doi.org/10.1016/j.apenergy.2016.08.165"
    KALTSCH_2020 = "https://doi.org/10.1007/978-3-662-61190-6"
    MTA = "MTA Galaxy tech GLT 210/SSN in https://www.mta.de/fileadmin/user_upload/MTA-Produktuebersicht-Prozesskuehlung.pdf"
    Mathiesen_2015 = "https://doi.org/10.1016/j.apenergy.2015.01.075"
    PLAN_BIOGAS = "https://planet-biogas.de/biogasprodukte/bhkw/"
    PVMAG_2020 = "https://perma.cc/CFH9-98J5"
    Redondo_2016 = "https://doi.org/10.1109%2FVPPC.2016.7791723"
    SMT_2018 = "https://perma.cc/FF2E-SE33"
    VDI2067 = "https://www.beuth.de/de/technische-regel/vdi-2067-blatt-1/151420393"
    VIESSMANN = "https://perma.cc/U2JM-R2L7"
    Vartiainen_2019 = "https://doi.org/10.1002/pip.3189"
    Wang_2018 = "https://doi.org/10.1016/j.apenergy.2018.08.124"
    Wolf_2017 = "https://doi.org/10.18419/opus-9593"
    # SORTING_END

class DataBase:
    from draf.conventions import Descs
    from draf.prep import param_funcs as funcs

    # SORTING_START
    c_EEG_ = ParDat(name="c_EEG_", data=0.065, doc=f"EEG levy", src=SRC.BMWI_2020, unit="€/kWh_el")
    c_Fuel_F = ParDat(name="c_Fuel_F", data=pd.Series({"ng": 0.04, "bio": 0.02}), doc="Fuel cost", unit="€/kWh")
    c_Fuel_co2_ = ParDat(name="c_Fuel_co2_", data=55, doc="CO2 price for non-electricity", unit="€/tCO2eq")
    c_P2H_inv_ = ParDat(name="c_P2H_inv_", data=100, doc=f"System CAPEX.", src=SRC.SMT_2018, unit="€/kW_th")
    ce_Fuel_F = ParDat(name="ce_Fuel_F", data=pd.Series({"ng": 0.240, "bio": 0.075}), doc=f"Fuel carbon emissions.", src=f"{SRC.GEG9_2020} documentation in {SRC.IFEU_2008}", unit="kgCO2eq/kWh")
    cop_CM_ = ParDat(name="cop_CM_", data=3.98, doc="Coefficient of performance", src=SRC.MTA, unit="kWh_th/kW_el")
    eta_BES_cycle_ = ParDat(name="eta_BES_cycle_", data=0.95, doc="Cycling efficiency", src=SRC.Carroquino_2021)
    eta_BES_time_ = ParDat(name="eta_BES_time_", data=1 - (0.0035 + 0.024) / 2 / (30 * 24), doc="Efficiency due to self-discharge rate", src=SRC.Redondo_2016)  # "0.35% to 2.5% per month depending on state of charge"
    eta_CHP_el_F = funcs.eta_CHP_el_F()
    eta_CHP_th_F = funcs.eta_CHP_th_F()
    eta_HP_ = ParDat(name="eta_HP_", data=0.5, doc=f"Ratio of reaching the ideal COP", src=SRC.KALTSCH_2020)
    eta_P2H_ = ParDat(name="eta_P2H_",data=0.9, doc="Efficiency", src=SRC.Wang_2018)
    eta_TES_in_ = ParDat(name="eta_TES_in_", data=0.99, doc=f"Loading efficiency", src=SRC.FFE_2016)
    eta_TES_time_ = ParDat(name="eta_TES_time_", data=0.95, doc=f"Storing efficiency", src=SRC.FFE_2016)
    k_BES_RMI_ = ParDat(name="k_BES_RMI_", data=0.02, doc=Descs.RMI.en, src=SRC.Juelch_2016)
    k_BES_inPerCapa_ = ParDat(name="k_BES_inPerCapa_", data=0.7, doc=f"Ratio charging power / capacity", src=SRC.Figgener_2021)
    k_BES_outPerCapa_ = ParDat(name="k_BES_outPerCapa_", data=0.7, doc=f"Ratio discharging power / capacity", src=SRC.Figgener_2021)
    k_CHP_RMI_ = ParDat(name="k_CHP_RMI_", data=0.06 + 0.02, doc=Descs.RMI.en, src=SRC.VDI2067)
    k_CM_RMI_ = ParDat(name="k_CM_RMI_", data=0.01 + 0.015, doc=Descs.RMI.en, src=SRC.VDI2067)
    k_HOB_RMI_ = ParDat(name="k_HOB_RMI_", data=0.04, doc=Descs.RMI.en)  # TODO
    k_HP_RMI_ = ParDat(name="k_HP_RMI_", data=0.01 + 0.015, doc=Descs.RMI.en, src=SRC.VDI2067)
    k_PV_RMI_ = ParDat(name="k_PV_RMI_", data=0.02, doc=Descs.RMI.en, src=SRC.ISE_2018)
    k_TES_RMI_ = ParDat(name="k_TES_RMI_", data=0.001, doc=Descs.RMI.en, src=SRC.FFE_2016)
    ol_BES_ = ParDat(name="ol_BES_", data=20, doc=f"Operation life", src=SRC.Juelch_2016, unit="a")
    ol_CHP_ = ParDat(name="ol_CHP_", data=15, doc=f"Operation life", src=SRC.VDI2067, unit="a")
    ol_HP_ = ParDat(name="ol_HP_", data=18, doc=f"Operation life", src=SRC.VDI2067, unit="a")
    ol_PV_ = ParDat(name="ol_PV_", data=25, doc=f"Operation life", src=SRC.ISE_2018, unit="a")
    ol_TES_ = ParDat(name="ol_TES_", data=30, doc=f"Operation life", src=SRC.BRACCO_2016, unit="a")
    # SORTING_END

# fmt: on
