#!/bin/sh

# format and sort section

black draf tests
isort draf tests
python draf/sort_sections.py