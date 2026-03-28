This is the implementation of the paper “'Integrating machine learning and molecular dynamics for predicting mechanical properties of MXene/Ni composites'.

## Dataset
Laser: https://docs.google.com/spreadsheets/d/183KgvmUCT79BmPexeQbsxTLjefvmwUuR

Tension: https://docs.google.com/spreadsheets/d/1RVRcfM41Z7x-oZpf02Zxjg7e-N5db7KK

Polishing: https://docs.google.com/spreadsheets/d/1EHPqVT6uOG64yJjCdtRLqy4fdTvQliet/edit?gid=2111871208#gid=2111871208

## How to run?

Developed with PyCharm.

```bash
git clone <repo-url>
cd <project-folder>
bash bash.sh
```

The `bash.sh` script:

* Upgrades `pip`
* Installs all dependencies from `requirements.txt`

---

## Experiments

Supported models:

* **XGBRegressor**
* **LightGBM**
* **VARMAX (VARX)**
* **Ridge-AR**

---

### Select dataset and model

All experiments are conducted by running `src/main.py`. 
First, choose a dataset, for example:
```python
DATA_TYPE = "tension"     # tension | polishing | laser
```
then choose a models, for example:
```python
METHOD_NAME = "XGBRegressor" # XGBRegressor | LightGBM | VARX | RIDGE-AR | ...
```
---

### Output

Results are saved in:

```
out/<METHOD>_<DATASET>_<TIMESTAMP>/
```

Our experiment is saved in the folder ``result``, for example:
``result/result (LightGBM)/laser/20260215``: the result of LightBGM on the laser dataset. 