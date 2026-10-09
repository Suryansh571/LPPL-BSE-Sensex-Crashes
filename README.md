# LPPL-BSE-Sensex-Crashes

Reproducibility repository for the paper:

"Nonlinear Modelling of Financial Crashes: An Empirical Examination of Log-Periodic Power Law Signatures in Indian Equity Markets"

## Authors

Tithi Mishra, Rajesh Mahadeva, Suryansh Sunil, Amit Kumar Goyal, and Varun Sarda

## Contents

- `LPPL_fit_multi_start.py` — joint nonlinear search over all seven LPPL parameters, 40 random starts, sensitivity plots, and diagnostics
- `LPPL_fit_filimonov.py` — same search after the Filimonov and Sornette (2013) reduction: the three linear parameters are solved by least squares at each trial of `(tc, beta, omega, phi)`
- Five CSV files with the BSE Sensex windows used for the five crashes (1990–2020)

## How to run

```bash
pip install pandas numpy matplotlib scipy openpyxl
python LPPL_fit_multi_start.py
python LPPL_fit_filimonov.py
