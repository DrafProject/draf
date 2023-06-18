#!/bin/sh

# format and sort section

black draf tests examples
isort draf tests examples
python draf/sort_sections.py
