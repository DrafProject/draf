from draf.models import der_hut, investment, minimal_example, mo_bes, mo_bes_2, p1


def test_models():
    der_hut.main()
    minimal_example.main()
    mo_bes.main()
    investment.main()
    mo_bes_2.main()
    p1.main()  # TODO: find the bug
