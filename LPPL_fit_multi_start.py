import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import f
import warnings
import os
warnings.filterwarnings("ignore")

# ====================== SETTINGS ======================
OUTPUT_FOLDER = "LPPL_Results/"
FILE_PATH = "Crash_Window_1_02_01_1986_to_09_10_1990.csv"
CRASH_NAME = "Crash 1"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def lppl(t, A, B, C, tc, beta, omega, phi):
    return A + B * (tc - t)**beta * (1 + C * np.cos(omega * np.log(tc - t) + phi))

def lppl_loss(params, t, y):
    try:
        return np.sum((y - lppl(t, *params))**2)
    except:
        return 1e10

def fit_single_crash():
    df = pd.read_csv(FILE_PATH)
    df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)

    t = np.array([(date - df['Date'].iloc[0]).days for date in df['Date']])
    y = np.log(df['Close'].values)

    bounds = [(0, None), (None, None), (-0.5, 0.5), (t.max(), t.max()+400),
              (0.05, 0.95), (4.0, 15.0), (0, 2*np.pi)]

    best_res = None
    best_fun = np.inf
    np.random.seed(42)

    for i in range(40):
        init = [float(y.mean()), -0.1, np.random.uniform(-0.4, 0.4),
                float(t.max() + np.random.uniform(15, 80)),
                np.random.uniform(0.1, 0.85), np.random.uniform(6, 12),
                np.random.uniform(0, 2*np.pi)]

        res = minimize(lppl_loss, init, args=(t, y), bounds=bounds,
                       method='L-BFGS-B', options={'maxiter': 8000})

        if res.fun < best_fun:
            best_fun = res.fun
            best_res = res

    params = best_res.x
    A, B, C, tc_opt, beta_opt, omega_opt, phi_opt = params
    rmse = np.sqrt(best_fun / len(t))
    y_pred = lppl(t, *params)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    # ====================== AIC & BIC ======================
    n = len(t)
    k_full = 7
    aic = n * np.log(best_fun / n) + 2 * k_full
    bic = n * np.log(best_fun / n) + k_full * np.log(n)

    # Comparison against the pure power law (C = 0). Not a one-degree nested F-test:
    # omega and phi drop out when C = 0.
    def reduced(p, t, y):
        return np.sum((y - (p[0] + p[1]*(p[2]-t)**p[3]))**2)
    res_red = minimize(reduced, [y.mean(), -0.1, tc_opt, beta_opt], args=(t, y))
    F = ((res_red.fun - best_fun)/1) / (best_fun / (len(t)-7)) if best_fun > 0 else 0
    f_pvalue = 1 - f.cdf(F, 1, len(t)-7) if F > 0 else 0.0

    # One calendar rule: date is the integer part of tc; days are the gap to the peak.
    estimated_date = df['Date'].iloc[0] + pd.to_timedelta(int(np.floor(tc_opt)), unit='D')
    days_after = int((estimated_date.normalize() - df['Date'].iloc[-1].normalize()).days)

    print(f"\n=== {CRASH_NAME} ===")
    print(f"Sample end: {df['Date'].iloc[-1].date()}")
    print(f"Estimated Date: {estimated_date.date()}")
    print(f"Days after sample end = {days_after}")
    print(f"tc (raw) = {tc_opt:.4f}")
    print(f"A = {A:.4f} | B = {B:.4f}")
    print(f"beta = {beta_opt:.3f} | omega = {omega_opt:.3f} | phi = {phi_opt:.3f} | C = {C:.3f}")
    print(f"RMSE = {rmse:.4f} | R2 = {r2:.4f} | AIC = {aic:.2f} | BIC = {bic:.2f} | F p-value = {f_pvalue:.4g}")

    # ====================== SAVE EXCEL ======================
    results_df = pd.DataFrame({
        'Crash': [CRASH_NAME],
        'Sample_end': [df['Date'].iloc[-1].date()],
        'Estimated_Date': [estimated_date.date()],
        'Days_after_sample_end': [days_after],
        'tc_days': [round(tc_opt, 1)],
        'A': [round(A, 4)],
        'B': [round(B, 4)],
        'beta': [round(beta_opt, 3)],
        'omega': [round(omega_opt, 3)],
        'phi': [round(phi_opt, 3)],
        'C': [round(C, 3)],
        'RMSE': [round(rmse, 4)],
        'R2': [round(r2, 4)],
        'AIC': [round(aic, 2)],
        'BIC': [round(bic, 2)],
        'F_pvalue': [f_pvalue]
    })

    excel_path = f"{OUTPUT_FOLDER}{CRASH_NAME.replace(' ', '_')}_Results.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"Excel saved: {excel_path}")

    # ====================== MAIN LPPL FIT GRAPH ======================
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], 'b-', label='Actual Sensex')
    plt.plot(df['Date'], np.exp(y_pred), 'r--', label='LPPL Fit')
    plt.axvline(estimated_date, color='black', linestyle=':',
                label=f'Estimated tc, {days_after} days after sample end')
    plt.xlabel('Date')
    plt.ylabel('Sensex Close Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_path = f"{OUTPUT_FOLDER}{CRASH_NAME.replace(' ', '_')}_Fit.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"Main Fit Graph saved: {graph_path}")

    # ====================== SENSITIVITY ANALYSIS PLOTS ======================
    fixed = params.copy()

    def sens_rmse(idx, vals):
        rmses = []
        for v in vals:
            p = fixed.copy()
            p[idx] = v
            try:
                rmses.append(np.sqrt(lppl_loss(p, t, y)))
            except:
                rmses.append(np.nan)
        return np.array(rmses)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(np.linspace(4, 15, 150), sens_rmse(5, np.linspace(4, 15, 150)))
    axs[0, 0].axvline(omega_opt, color='red', linestyle='--')
    axs[0, 0].set_title('ω vs RMSE')
    axs[0, 0].set_xlabel('ω')
    axs[0, 0].set_ylabel('RMSE')

    axs[0, 1].plot(np.linspace(0, 2*np.pi, 150), sens_rmse(6, np.linspace(0, 2*np.pi, 150)))
    axs[0, 1].axvline(phi_opt, color='red', linestyle='--')
    axs[0, 1].set_title('ϕ vs RMSE')
    axs[0, 1].set_xlabel('ϕ')

    axs[1, 0].plot(np.linspace(tc_opt-100, tc_opt+100, 150), sens_rmse(3, np.linspace(tc_opt-100, tc_opt+100, 150)))
    axs[1, 0].axvline(tc_opt, color='red', linestyle='--')
    axs[1, 0].set_title('tc vs RMSE')
    axs[1, 0].set_xlabel('tc (days)')

    axs[1, 1].plot(np.linspace(0.05, 0.95, 150), sens_rmse(4, np.linspace(0.05, 0.95, 150)))
    axs[1, 1].axvline(beta_opt, color='red', linestyle='--')
    axs[1, 1].set_title('β vs RMSE')
    axs[1, 1].set_xlabel('β')

    plt.suptitle(f'Sensitivity of RMSE to LPPL Parameters - {CRASH_NAME}')
    plt.tight_layout()

    sens_path = f"{OUTPUT_FOLDER}{CRASH_NAME.replace(' ', '_')}_Sensitivity.png"
    plt.savefig(sens_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity Plot saved: {sens_path}\n")

# ====================== RUN ======================
fit_single_crash()