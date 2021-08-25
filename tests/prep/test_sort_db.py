from draf.prep.sort_db import sort_sections

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
    c_EEG_ = ParDat(name="c_EEG_", data=0.065)
"""


c = """\
    ol_PV_ = ParDat(name="ol_PV_", data=25)
"""

d = """\
    # SORTING_END

e
"""


def test_sort_sections():
    assert a + b + c + d == sort_sections(a + c + b + d)
