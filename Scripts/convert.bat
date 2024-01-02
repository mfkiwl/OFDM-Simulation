:: This script convert all the eps file into pdf
@echo off
cd ..
for %%f in (Source\results\*.eps) do epstopdf %%f