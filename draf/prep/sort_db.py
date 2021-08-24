def sort_sections(s):
    new = ""
    starter = "\n    # SORTING_START\n"
    ender = "    # SORTING_END\n"

    whole_string = s.split(starter)
    new += whole_string[0]
    rest = whole_string[1:]

    for part in rest:
        x = part.split(ender)
        to_sort = x[0]
        new += starter[:-1]
        sorted_list = sorted(to_sort.split("\n"))
        sorted_string = "\n".join(sorted_list)
        new += sorted_string
        new += "\n" + ender

        new += x[1]

    return new


if __name__ == "__main__":
    from pathlib import Path

    fp = Path(__file__).parent / "data_base.py"
    fp.write_text(sort_sections(fp.read_text()))
