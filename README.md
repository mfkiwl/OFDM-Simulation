# References
1. https://github.com/kornthum/OFDM_64QAM_Python
2. https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html

# To Convert notebook to pdf
```
cd Source
jupyter nbconvert --to pdf 64_QAM_OFDM.ipynb
```

# To Convert notebook to markdown
```
cd Source
jupyter nbconvert --to markdown 64_QAM_OFDM.ipynb
```

# To convert markdown to latex and then into pdf
```
pandoc --listings -f markdown -t latex 64_QAM_OFDM.md -o 64_QAM_OFDM.tex
pdflatex 64_QAM_OFDM.tex
```