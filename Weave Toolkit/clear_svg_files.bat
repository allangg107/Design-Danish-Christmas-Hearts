@echo off
echo This script will delete all .svg files in the current directory.
echo.

REM Count how many .svg files exist in the current directory
set count=0
for %%F in (*.svg) do set /a count+=1

if %count% EQU 0 (
    echo No SVG files found in the current directory.
) else (
    echo Found %count% SVG files in the current directory.
    del /q *.svg
    echo All SVG files have been deleted.
)

echo.