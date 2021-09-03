"""A small script using the model pv_bes."""


def main():
    import draf
    from draf.models.gp import pv_bes as mod

    cs = draf.CaseStudy(name="min_imp", year=2019, freq="60min", country="DE", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 2)
    sc = cs.add_REF_scen(doc="no BES").set_params(mod.params_func)
    sc.update_params(P_PV_CAPx_=100, c_EL_buyPeak_=50)
    cs.add_scens(nParetoPoints=2)
    cs.set_model(mod.model_func).optimize(postprocess_func=mod.postprocess_func)
