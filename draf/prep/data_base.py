import pandas as pd

from draf.prep.par_dat import ParDat, Source

# fmt: off

# This file is ignored from the black formatter.
# Data entries are alphabetically sorted within the formatting routine.


class SRC:
    # SORTING_START
    Arat_2017 =        Source(url="https://doi.org/10.1016/j.energy.2016.11.073", doc="Fig. 3", bib=r"@article{Arat_2017, doi={10.1016/j.energy.2016.11.073}, url={https://doi.org/10.1016/j.energy.2016.11.073}, year={2017}, month=jan, publisher={Elsevier {BV}}, volume={119}, pages={1159--1170}, author={Halit Arat and Oguz Arslan}, title={Exergoeconomic analysis of district heating system boosted by the geothermal heat pump}, journal={Energy}}")
    ASUE_2011 =        Source(url="https://perma.cc/KHG2-WPPX", doc="p.12", bib=r"@www{ASUE_2011, author={ASUE}, title={{BHKW-Kenndaten 2011}}, url={https://perma.cc/KHG2-WPPX}, urldate={03.09.2021}}")
    BMWI_2020 =        Source(url="https://www.bmwi-energiewende.de/EWD/Redaktion/Newsletter/2020/11/Meldung/News1.html", bib=r"@www{BMWI_2020, author={BMWI}, title={{EEG-Umlage sinkt 2021}}, url={https://www.bmwi-energiewende.de/EWD/Redaktion/Newsletter/2020/11/Meldung/News1.html}, urldate={03.09.2021}}")
    Bracco_2016 =      Source(url="https://doi.org/10.1016/j.energy.2016.01.050", bib=r"@article{Bracco_2016, author={Bracco, Stefano and Dentici, Gabriele and Siri, Silvia}, year={2016}, title={{DESOD: a mathematical programming tool to optimally design a distributed energy system}}, keywords={Cogeneration;Distributed generation;Optimal design;Renewables;Storage;Trigeneration}, pages={298--309}, volume={100}, issn={03605442}, journal={{Energy}}, doi={10.1016/j.energy.2016.01.050}}")
    Carroquino_2021 =  Source(url="https://doi.org/10.3390/app11083587", bib=r"@article{Carroquino_2021, author={Javier Carroquino and Cristina Escriche-Mart{\'{\i}}nez and Luis Vali{\~{n}}o and Rodolfo Dufo-L{\'{o}}pez}, doi={10.3390/app11083587}, journal={Applied Sciences}, month={apr}, number={8}, pages={3587}, publisher={{MDPI} {AG}}, title={Comparison of Economic Performance of Lead-Acid and Li-Ion Batteries in Standalone Photovoltaic Energy Systems}, url={https://doi.org/10.3390%2Fapp11083587}, volume={11}, year=2021}")
    DanishEA_2022 =    Source(url="https://ens.dk/en/our-services/projections-and-models/technology-data/technology-data-generation-electricity-and")
    Estoolbox =        Source(url="https://estoolbox.org/index.php/en/background-2/8-samples/8-tes-introduction")
    FFE_2016 =         Source(url="https://perma.cc/F9X5-HG3B", bib=r"@www{FFE_2016, author={FFE}, title={{Verbundforschungsvorhaben Merit Order der Energiespeicherung im Jahr 2030}}, url={https://perma.cc/F9X5-HG3B}, urldate={03.09.2021}}")
    Figgener_2021 =    Source(url="https://doi.org/10.1016/j.est.2020.101982", bib=r"@article{Figgener_2021, author={Jan Figgener and Peter Stenzel and Kai-Philipp Kairies and Jochen Lin{\ss}en and David Haberschusz and Oliver Wessels and Martin Robinius and Detlef Stolten and Dirk Uwe Sauer}, doi={10.1016/j.est.2020.101982}, journal={Journal of Energy Storage}, month={jan}, pages={101982}, publisher={Elsevier {BV}}, title={The development of stationary battery storage systems in Germany {\textendash} status 2020}, url={https://doi.org/10.1016%2Fj.est.2020.101982}, volume={33}, year=2021}")
    GEG9_2020 =        Source(url="https://www.buzer.de/Anlage_9_GEG.htm", doc="documentation in IFEU_2008", bib=r"@www{GEG9_2020, author={buzer}, title={{Anlage 9 (zu § 85 Absatz 6) Umrechnung in Treibhausgasemissionen}}, url={https://www.buzer.de/Anlage_9_GEG.htm}, urldate={03.09.2021}}")
    IFEU_2008 =        Source(url="https://perma.cc/UZX6-ZN73", bib=r"@www{IFEU_2008, author={ifeu}, title={{Basisdaten zu THG-Bilanzen für Biogas-Prozessketten und Erstellung neuer THG-Bilanzen}}, url={https://perma.cc/UZX6-ZN73}, urldate={03.09.2021}}")
    ISE_2018 =         Source(url="https://perma.cc/MQS2-DCPL", bib=r"@www{ISE_2018, author={Fraunhofer ISE}, title={{Stromgestehungskosten Erneuerbare Energien}}, url={https://perma.cc/MQS2-DCPL}, urldate={03.09.2021}}")
    Juelch_2016 =      Source(url="https://doi.org/10.1016/j.apenergy.2016.08.165", bib=r"@article{Juelch_2016, author={Verena J\"{u}lch}, doi={10.1016/j.apenergy.2016.08.165}, journal={Applied Energy}, month={dec}, pages={1594--1606}, publisher={Elsevier {BV}}, title={Comparison of electricity storage options using levelized cost of storage ({LCOS}) method}, url={https://doi.org/10.1016/j.apenergy.2016.08.165}, volume={183}, year={2016}}")
    KALTSCH_2020 =     Source(url="https://doi.org/10.1007/978-3-662-61190-6")
    Mathiesen_2015 =   Source(url="https://doi.org/10.1016/j.apenergy.2015.01.075", bib=r"@article{Mathiesen_2015, author={B.V. Mathiesen and H. Lund and D. Connolly and H. Wenzel and P.A. Østergaard and B. Möller and S. Nielsen and I. Ridjan and P. Karnøe and K. Sperling and F.K. Hvelplund}, doi={10.1016/j.apenergy.2015.01.075}, journal={Applied Energy}, month={may}, pages={139--154}, publisher={Elsevier {BV}}, title={Smart Energy Systems for coherent 100{\%} renewable energy and transport solutions}, url={https://doi.org/10.1016%2Fj.apenergy.2015.01.075}, volume={145}, year= 2015}")
    MTA =              Source(url="https://www.mta.de/fileadmin/user_upload/MTA-Produktuebersicht-Prozesskuehlung.pdf", doc="MTA Galaxy tech GLT 210/SSN")
    nProParams =       Source(url="https://www.npro.energy/main/en/help/economic-parameters", doc="url-date: 2022-10-11")
    Petkov_2020 =      Source(url="https://doi.org/10.1016/j.apenergy.2020.115197", doc="Page 7", bib=r"@article{Petkov2020, doi={10.1016/j.apenergy.2020.115197}, year={2020}, month=sep, publisher={Elsevier {BV}}, volume={274}, pages={115197}, author={Ivalin Petkov and Paolo Gabrielli}, title={Power-to-hydrogen as seasonal energy storage: an uncertainty analysis for optimal design of low-carbon multi-energy systems}, journal={Applied Energy}}")
    PLAN_BIOGAS =      Source(url="https://planet-biogas.de/biogasprodukte/bhkw/", bib=r"@www{PLAN_BIOGAS, author={PlanET}, title={{PLAN_BIOGAS Biogas-BHKW: mehr Wirkungsgrad senkt die Kosten}}, url={https://planet-biogas.de/biogasprodukte/bhkw/}, urldate={03.09.2021}}")
    PVMAG_2020 =       Source(url="https://perma.cc/CFH9-98J5", bib=r"@www{PVMAG_2020, author={pv magazine}, title={{pv magazine Marktübersicht für Großspeicher aktualisiert}}, url={https://perma.cc/CFH9-98J5}, urldate={03.09.2021}}")
    Redondo_2016 =     Source(url="https://doi.org/10.1109%2FVPPC.2016.7791723", bib=r"@inproceedings{Redondo_2016, author={Eduardo Redondo-Iglesias and Pascal Venet and Serge Pelissier}, booktitle={2016 {IEEE} Vehicle Power and Propulsion Conference ({VPPC})}, doi={10.1109/vppc.2016.7791723}, month={oct}, publisher={{IEEE}}, title={Measuring Reversible and Irreversible Capacity Losses on Lithium-Ion Batteries}, url={https://doi.org/10.1109%2Fvppc.2016.7791723}, year=2016}")
    SMT_2018 =         Source(url="https://perma.cc/FF2E-SE33", bib=r"@www{SMT_2018, author={Hinterberger, Robert and Hinrichsen, Johannes and Dedeyne, Stefanie}, title={{Power-To-Heat Anlagen zur Verwertung von EEÜberschussstrom – neuer Rechtsrahmen im Energiewirtschaftsgesetz, bisher ohne Wirkung}}, url={https://perma.cc/FF2E-SE33}, urldate={03.09.2021}}")
    Tieman_2020 =      Source(url="https://doi.org/10.1016/j.enconman.2020.112539", bib=r"@article{Tieman_2020, doi={10.1016/j.enconman.2020.112539}, year={2020}, month={mar}, publisher={Elsevier {BV}}, volume={208}, pages={112539}, author={Paul Hendrik Tiemann and Astrid Bensmann and Volker Stuke and Richard Hanke-Rauschenbach}, title={Electrical energy storage for industrial grid fee reduction {\textendash} A large scale analysis}}")
    Vartiainen2021 =   Source(url="https://doi.org/10.1002/solr.202100487", bib=r"@article{Vartiainen2021, author = {Vartiainen, Eero and Breyer, Christian and Moser, David and {Rom{\'{a}}n Medina}, Eduardo and Busto, Chiara and Masson, Ga{\"{e}}tan and Bosch, Elina and J{\"{a}}ger-Waldau, Arnulf}, doi = {10.1002/solr.202100487}, issn = {2367198X}, journal = {Solar RRL}, title = {{True Cost of Solar Hydrogen}}, year = {2021}}")
    Vartiainen_2019 =  Source(url="https://doi.org/10.1002/pip.3189", bib=r"@article{Vartiainen_2019, author={Eero Vartiainen and GaÃ«tan Masson and Christian Breyer and David Moser and Eduardo Rom{\'{a}}n Medina}, doi={10.1002/pip.3189}, journal={Progress in Photovoltaics: Research and Applications}, month={aug}, number={6}, pages={439--453}, publisher={Wiley}, title={Impact of weighted average cost of capital, capital expenditure, and other parameters on future utility-scale {PV} levelised cost of electricity}, url={https://doi.org/10.1002%2Fpip.3189}, volume={28}, year= 2019}")
    VDI2067 =          Source(url="https://www.beuth.de/de/technische-regel/vdi-2067-blatt-1/151420393", bib=r"@www{VDI2067, author={Beuth}, title={{VDI 2067 Blatt 1:2012-09}}, url={https://www.beuth.de/de/technische-regel/vdi-2067-blatt-1/151420393}, urldate={03.09.2021}}")
    VIESSMANN =        Source(url="https://perma.cc/U2JM-R2L7", bib=r"@www{VIESSMANN, author={Viessmann}, title={{Preisliste DE Heizsysteme}}, url={https://perma.cc/U2JM-R2L7}, urldate={03.09.2021}}")
    Wang_2018 =        Source(url="https://doi.org/10.1016/j.apenergy.2018.08.124", bib=r"@article{Wang_2018, author={Xuan Wang and Ming Jin and Wei Feng and Gequn Shu and Hua Tian and Youcai Liang}, doi={10.1016/j.apenergy.2018.08.124}, journal={Applied Energy}, month={nov}, pages={679--695}, publisher={Elsevier {BV}}, title={Cascade energy optimization for waste heat recovery in distributed energy systems}, url={https://doi.org/10.1016%2Fj.apenergy.2018.08.124}, volume={230}, year= 2018}")
    Weber_2008 =       Source(url="https://doi.org/10.5075/epfl-thesis-4018")
    Wolf_2017 =        Source(url="https://doi.org/10.18419/opus-9593", bib=r"@misc{Wolf_2017, author={Wolf, Stefan}, doi={10.18419/OPUS-9593}, keywords={620}, language={de}, publisher={Universität Stuttgart}, title={Integration von Wärmepumpen in industrielle Produktionssysteme : Potenziale und Instrumente zur Potenzialerschließung}, url={http://elib.uni-stuttgart.de/handle/11682/9610}, year={2017}}")
    # SORTING_END


class DataBase:
    from draf.conventions import Descs, Etypes
    from draf.prep import param_funcs as funcs

    # SORTING_START
    ce_Fuel_F = ParDat(name="ce_Fuel_F", data=pd.Series({"ng": 0.240, "bio": 0.075}), doc="Fuel carbon emissions.", src="@GEG9_2020", unit="kgCO2eq/kWh")
    cop_CM_ = ParDat(name="cop_CM_", data=3.98, doc="Coefficient of performance", src="@MTA", unit="kWh_th/kW_el")
    c_EG_EEG_ = ParDat(name="c_EG_EEG_", data=0.065, doc="EEG levy", src="@BMWI_2020", unit="€/kWh_el")
    c_Elc_inv_ = ParDat(name="c_Elc_inv_", data=1295, doc="CAPEX", src="@Petkov_2020", unit="€/kW_el")
    c_FC_inv_ = ParDat(name="c_FC_inv_", data=1684, doc="CAPEX", src="@Petkov_2020", unit="€/kW_el")
    c_Fuel_co2_ = ParDat(name="c_Fuel_co2_", data=55, doc="CO2 price for non-electricity", unit="€/tCO2eq")
    c_Fuel_F = ParDat(name="c_Fuel_F", data=pd.Series({"ng": 0.04, "bio": 0.02}), doc="Fuel cost", unit="€/kWh")
    c_H2S_inv_ = ParDat(name="c_H2S_inv_", data=10, doc="CAPEX", src="@Petkov_2020", unit="€/kWh")
    c_P2H_inv_ = ParDat(name="c_P2H_inv_", data=100, doc="CAPEX", src="@SMT_2018", unit="€/kW_th")
    eta_BES_cycle_ = ParDat(name="eta_BES_cycle_", data=0.95, doc="Cycling efficiency", src="@Carroquino_2021")
    eta_BES_self_ = ParDat(name="eta_BES_self_", data=(0.0035 + 0.024) / 2 / (30 * 24), doc="Self-discharge rate per hour", src="@Redondo_2016")  # "0.35% to 2.5% per month depending on state of charge"
    eta_CHP_el_F = funcs.eta_CHP_el_F()
    eta_CHP_th_F = funcs.eta_CHP_th_F()
    eta_Elc_ = ParDat(name="eta_Elc_", data=0.71, doc="Efficiency", src="@Petkov_2020")
    eta_FC_el_ = ParDat(name="eta_FC_el_", data=0.50, doc="Electrical efficiency", src="@Petkov_2020")
    eta_FC_th_ = ParDat(name="eta_FC_th_", data=0.34, doc="Thermal efficiency", src="@Petkov_2020")
    eta_H2S_ch_ = ParDat(name="eta_H2S_ch_", data=0.90**0.5, doc="Charging efficiency", src="@Petkov_2020")  # 90% cycling efficiency
    eta_H2S_dis_ = ParDat(name="eta_H2S_dis_", data=0.90**0.5, doc="Discharging efficiency", src="@Petkov_2020")  # 90% cycling efficiency
    eta_HOB_ = ParDat(name="eta_HOB_", data=0.9, doc="Thermal efficiency", unit="kWh_th/kWh", src="@Weber_2008")
    eta_HP_ = ParDat(name="eta_HP_", data=0.5, doc="Ratio of reaching the ideal COP", src="@KALTSCH_2020")
    eta_P2H_ = ParDat(name="eta_P2H_", data=0.9, doc="Efficiency", src="@Wang_2018")
    eta_TES_in_ = ParDat(name="eta_TES_in_", data=0.99, doc="Loading efficiency", src="@FFE_2016")
    eta_TES_self_ = ParDat(name="eta_TES_self_", data=0.00029167, doc="Self-discharge", src="@Estoolbox")
    k_BES_inPerCap_ = ParDat(name="k_BES_inPerCap_", data=0.7, doc="Maximum charging power per capacity", src="@Figgener_2021")
    k_BES_outPerCap_ = ParDat(name="k_BES_outPerCap_", data=0.7, doc="Maximum discharging power per capacity", src="@Figgener_2021")
    k_BES_RMI_ = ParDat(name="k_BES_RMI_", data=0.02, doc=Descs.RMI.en, src="@Juelch_2016")
    k_CHP_RMI_ = ParDat(name="k_CHP_RMI_", data=0.18, doc=Descs.RMI.en, src="@Weber_2008")
    k_CM_RMI_ = ParDat(name="k_CM_RMI_", data=0.01 + 0.015, doc=Descs.RMI.en, src="@VDI2067")
    k_Elc_RMI_ = ParDat(name="k_Elc_RMI_", data=0.038, src="@Petkov_2020")
    k_FC_RMI_ = ParDat(name="k_FC_RMI_", data=0.038, doc=Descs.RMI.en, src="@Petkov_2020")
    k_H2S_RMI_ = ParDat(name="k_H2S_RMI_", data=0.023, doc=Descs.RMI.en, src="@Petkov_2020")
    k_HOB_RMI_ = ParDat(name="k_HOB_RMI_", data=0.18, doc=Descs.RMI.en, src="@Weber_2008")
    k_HP_RMI_ = ParDat(name="k_HP_RMI_", data=0.01 + 0.015, doc=Descs.RMI.en, src="@VDI2067")  # 5% assumed by Cox_2022 (https://doi.org/10.1016/j.apenergy.2021.118499)
    k_PV_RMI_ = ParDat(name="k_PV_RMI_", data=0.02, doc=Descs.RMI.en, src="@ISE_2018")
    k_TES_RMI_ = ParDat(name="k_TES_RMI_", data=0.001, doc=Descs.RMI.en, src="@FFE_2016")
    N_BES_ = ParDat(name="N_BES_", data=20, doc=Etypes.N.en, src="@Juelch_2016", unit="a")
    N_CHP_ = ParDat(name="N_CHP_", data=25, doc=Etypes.N.en, src="@Weber_2008", unit="a")
    N_Elc_ = ParDat(name="N_Elc_", data=14, doc=Etypes.N.en, src="@Petkov_2020", unit="a")
    N_FC_ = ParDat(name="N_FC_", data=14, doc=Etypes.N.en, src="@Petkov_2020", unit="a")
    N_H2S_ = ParDat(name="N_H2S_", data=23, doc=Etypes.N.en, src="@Petkov_2020", unit="a")
    N_HOB_ = ParDat(name="N_HOB_", data=15, doc=Etypes.N.en, src="@Weber_2008", unit="a")
    N_HP_ = ParDat(name="N_HP_", data=18, doc=Etypes.N.en, src="@VDI2067", unit="a")
    N_P2H_ = ParDat(name="N_P2H_", data=30, doc=Etypes.N.en, unit="a")  # TODO find src
    N_PV_ = ParDat(name="N_PV_", data=25, doc=Etypes.N.en, src="@ISE_2018", unit="a")
    N_TES_ = ParDat(name="N_TES_", data=30, doc=Etypes.N.en, src="@Bracco_2016", unit="a")
    # SORTING_END

# fmt: on
