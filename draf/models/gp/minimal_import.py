"""A small script using using the model pv_bes."""


def main():
    import draf
    from draf.models.gp import pv_bes as mod

    cs = draf.CaseStudy(name="ShowCase", year=2017, freq="60min", country="DE")
    sc = cs.add_REF_scen(doc="no BES").set_params(mod.params_func)
    sc.update_params(P_PV_CAPx_=100, c_GRID_peak_=50)
    cs.add_scens(
        scen_vars=[
            ("c_GRID_T", "t", ["c_GRID_RTP_T", "c_GRID_TOU_T"]),
            ("E_BES_CAPx_", "b", [1000]),
        ],
        nParetoPoints=3,
    )
    cs.set_model(mod.model_func).optimize(postprocess_func=mod.postprocess_func).save()
