:: This script convert all the eps file into pdf
@echo off
for %%f in (Source\results\*.eps) do epstopdf %%f