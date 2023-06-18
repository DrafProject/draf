#!/bin/sh

# format and test with coverage report

clear
sh fmt.sh
pytest --cov-report=html
