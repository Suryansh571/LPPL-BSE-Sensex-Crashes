# LPPL-BSE-Sensex-Crashes

Reproducibility repository for the paper:

**"Nonlinear Modelling of Financial Crashes: An Empirical Examination of Log-Periodic Power Law Signatures in Indian Equity Markets"**

### Contents
- `lppl_fit_multi_start.py` — Main Python code with multi-start optimization (40 runs), sensitivity analysis, F-test, AIC/BIC, and plotting
- Five input CSV files containing BSE Sensex historical data windows for the five major crashes
- Generated output: Fit plots and sensitivity plots for each crash

### How to Run
```bash
pip install pandas numpy matplotlib scipy
python lppl_fit_multi_start.py
