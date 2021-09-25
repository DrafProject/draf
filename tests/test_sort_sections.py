from draf.sort_sections import sort_sections

a = """\
a

b
    # SORTING_START
    ASUE_2011 = "https://asue.de/sites"
    BMWI_2020 = "https://www.bmwi-energiewende.de/EWD"
    # SORTING_END

c:
    d

    # SORTING_START
"""

b = """\
    aa_EEG_ = ParDat(name="c_EG_EEG_", data=0.065)
"""


c = """\
    zz_PV_ = ParDat(name="ol_PV_", data=25)
"""

d = """\
    # SORTING_END

e
"""


def test_sort_sections():
    assert a + b + c + d == sort_sections(a + c + b + d)
