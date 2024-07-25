@echo off
setlocal
set PREV_PYTHONPATH = %PYTHONPATH%
set PYTHONPATH=.\;..\gym-fx;..\neat-test
python app/main.py %*
set PYTHONPATH=%PREV_PYTHONPATH%
endlocal 