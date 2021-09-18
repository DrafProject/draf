@REM format and fast-test
call fmt.bat
pytest -m="not slow"