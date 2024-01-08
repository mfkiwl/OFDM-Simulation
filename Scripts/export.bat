:: This script export all the draw.io files into pdf
@echo off
for %%f in (Report\Figures\*.drawio) do (
    draw.io -x -r --crop -f pdf %%f  
)