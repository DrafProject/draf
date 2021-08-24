def sort_lines_in_string(s: str) -> str:
    return "\n".join(sorted(s.split("\n")))


def sort_sections(s: str) -> str:
    starter = "\n    # SORTING_START\n"
    ender = "    # SORTING_END\n"

    whole_string = s.split(starter)
    new = whole_string[0]
    rest = whole_string[1:]

    for part in rest:
        x = part.split(ender)
        new += starter[:-1] + sort_lines_in_string(x[0]) + "\n" + ender + x[1]
    return new


if __name__ == "__main__":
    from pathlib import Path

    fp = Path(__file__).parent / "data_base.py"
    fp.write_text(sort_sections(fp.read_text()))
