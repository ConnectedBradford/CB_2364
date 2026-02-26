# CB_2364_ASQ3-EYFSP-Longitudinal-Analysis

![Python](https://img.shields.io/badge/python-3.11.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is the repo for paper "The association between the Ages and Stages Questionnaire 3 assessment at age 2 and the Early Years Foundation Stage at age 5: A longitudinal observational study using routine data"

## Pre-requisites

The statistical analyses and data processing reported in this study were conducted using Python 3.11 and the following core packages:

- pandas==1.5.3
- rpy2==3.6.4
- numpy==1.26.4
- statsmodels==0.14.5
- plotly==5.24.1
- tabulate==0.9.0
- rich==14.2.0
- IPython==8.31.0

All required package versions are specified in the provided
`environment.yml` file to ensure full reproducibility of the
computational environment.

The analysis environment can be recreated using Conda with:

```bash
conda env create -f environment.yml
conda activate asq3-eyfs
```

All analysis scripts are provided as Jupyter Notebook
(`.ipynb`) files and require execution within a Jupyter
Notebook-compatible environment (e.g. Jupyter Notebook or
JupyterLab).
