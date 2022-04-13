#!/bin/sh

# format and test with coverage report

sh fmt.sh
pytest --cov-report=html
