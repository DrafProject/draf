from draf.models import der_hut, investment, minimal, mo_bes, mo_bes_2, p1
from draf.models.pyo import minimal as pyo_minimal


def test_models():
    der_hut.main()
    minimal.main()
    mo_bes.main()
    investment.main()
    mo_bes_2.main()
    p1.main()

    pyo_minimal.main()
