import draf


def test_casestudy():
    cs = draf.CaseStudy()
    sc = cs.add_REF_scen()
    sc.add_dim("T", infer=True)
    sc.add_par("pv_capa_", 0, "", "kW_el")
    sc.add_var("C_", "total Costs", "k€/a")
    sc.prep.add_c_GRID_RTP_T()
    sc.prep.add_ce_GRID_T(method="XEF_PWL")
    sc.prep.add_ce_GRID_T(name="ce_GRID_MEF_T", method="MEF_PWL")


def test_minimal_example():
    from draf.models import minimal_example

    minimal_example.main()
