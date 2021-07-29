from draf.plotting import cs_plotting


def test_float_to_x():
    assert cs_plotting.float_to_int_to_string(2.4) == "2"
    assert cs_plotting.float_to_int_to_string(2.6) == "3"
    assert cs_plotting.float_to_string_with_precision_1(2.44) == "2.4"
    assert cs_plotting.float_to_string_with_precision_2(2.444) == "2.44"
