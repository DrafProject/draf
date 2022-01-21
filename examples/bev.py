import draf
from draf.components import *


def main():
    cs = draf.CaseStudy("bev", year=2019, freq="60min", coords=(49.01, 8.39))
    cs.set_time_horizon(start="Apr-01 00:00", steps=24 * 30)
    sc = cs.add_REF_scen(components=[eDem, EG, PV, BEV, Main])  # fmt: skip
    p_drive = sc.params.P_BEV_drive_TB.where(
        sc.params.P_BEV_drive_TB.index.get_level_values(0) % 24 < 12, 20
    )
    y_avail = (p_drive == 0).astype(int)
    sc.update_params(P_BEV_drive_TB=p_drive, y_BEV_avail_TB=y_avail)
    cs.optimize(logToConsole=False)
    return cs
