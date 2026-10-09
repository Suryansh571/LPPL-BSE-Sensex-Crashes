import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings("ignore")

# ====================== SETTINGS ======================
OUTPUT_FOLDER = "Filimonov_Results/"
FILE_PATH = "Crash_Window_1_02_01_1986_to_09_10_1990.csv"
CRASH_NAME = "Crash 1"
SHORT_WINDOW = False          # True = last 262 rows only
N_STARTS = 40
SEED = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_window(path, short=False):
    df = pd.read_csv(path)
    df["Close"] = df["Close"].astype(str).str.replace(",", "").astype(float)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[df["Close"] > 0].reset_index(drop=True)
    if short:
        df = df.tail(262).reset_index(drop=True)
    t = np.array([(d - df["Date"].iloc[0]).days for d in df["Date"]], dtype=float)
    y = np.log(df["Close"].values)
    return df, t, y

def linear_solve(t, y, tc, beta, omega):
    dt = tc - t
    if np.any(dt <= 1e-8) or not np.isfinite(beta):
        return None
    f = np.power(dt, beta)
    logdt = np.log(dt)
    X = np.column_stack([
        np.ones_like(t),
        f,
        f * np.cos(omega * logdt),
        f * np.sin(omega * logdt),
    ])
    if not np.all(np.isfinite(X)):
        return None
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 4:
        return None
    resid = y - X @ coef
    sse = float(np.dot(resid, resid))
    A, B, C1, C2 = map(float, coef)
    return A, B, C1, C2, sse

def recover_C_phi(B, C1, C2):
    if abs(B) < 1e-12:
        return np.nan, np.nan
    C = np.sqrt(C1**2 + C2**2) / abs(B)
    phi = np.arctan2(-C2 * np.sign(B), C1 * np.sign(B))
    if phi < 0:
        phi += 2 * np.pi
    return float(C), float(phi)

def loss_nonlinear(z, t, y):
    tc, beta, omega = z
    solved = linear_solve(t, y, tc, beta, omega)
    if solved is None:
        return 1e10
    return solved[4]

def fit_filimonov():
    df, t, y = load_window(FILE_PATH, SHORT_WINDOW)
    tmax = float(t.max())
    bounds = [(tmax, tmax + 400), (0.05, 0.95), (4.0, 15.0)]

    rng = np.random.default_rng(SEED)
    best = None
    n_finished = 0
    n_on_bound = 0
    losses = []

    for i in range(N_STARTS):
        z0 = [
            float(tmax + rng.uniform(15, 80)),
            float(rng.uniform(0.1, 0.85)),
            float(rng.uniform(6, 12)),
        ]
        res = minimize(
            loss_nonlinear, z0, args=(t, y),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 8000, "ftol": 1e-12},
        )
        if not np.isfinite(res.fun) or res.fun >= 1e9:
            continue
        n_finished += 1
        losses.append(res.fun)
        tc, beta, omega = res.x
        on_bound = (
            abs(beta - 0.05) < 1e-4 or abs(beta - 0.95) < 1e-4
            or abs(omega - 4.0) < 1e-4 or abs(omega - 15.0) < 1e-4
            or abs(tc - tmax) < 1e-3 or abs(tc - (tmax + 400)) < 1e-3
        )
        if on_bound:
            n_on_bound += 1
        if best is None or res.fun < best["sse"]:
            solved = linear_solve(t, y, tc, beta, omega)
            A, B, C1, C2, sse = solved
            C, phi = recover_C_phi(B, C1, C2)
            best = {
                "tc": tc, "beta": beta, "omega": omega,
                "A": A, "B": B, "C1": C1, "C2": C2,
                "C": C, "phi": phi, "sse": sse, "on_bound": on_bound,
            }

    if best is None:
        print("No successful start.")
        return

    tc = best["tc"]
    solved = linear_solve(t, y, tc, best["beta"], best["omega"])
    A, B, C1, C2, sse = solved
    yhat = (
        A
        + B * np.power(tc - t, best["beta"])
        + C1 * np.power(tc - t, best["beta"]) * np.cos(best["omega"] * np.log(tc - t))
        + C2 * np.power(tc - t, best["beta"]) * np.sin(best["omega"] * np.log(tc - t))
    )
    resid = y - yhat
    n = len(y)
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float(1 - sse / ss_tot) if ss_tot > 0 else np.nan
    sigma2 = sse / n
    k = 7
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)
    peak_date = df["Date"].iloc[-1]
    tc_date = peak_date + pd.Timedelta(days=float(tc - tmax))
    n_match = int(np.sum(np.abs(np.array(losses) - best["sse"]) < 1e-6))

    tag = "short262" if SHORT_WINDOW else "full"
    row = {
        "Crash": CRASH_NAME,
        "Window": tag,
        "Estimator": "Filimonov split",
        "Peak date": peak_date.date().isoformat(),
        "Estimated tc": tc_date.date().isoformat(),
        "Days after sample end": round(float(tc - tmax), 1),
        "tc days from window start": round(float(tc), 1),
        "A": best["A"],
        "B": best["B"],
        "C": best["C"],
        "C1": best["C1"],
        "C2": best["C2"],
        "beta": best["beta"],
        "omega": best["omega"],
        "phi": best["phi"],
        "RMSE": rmse,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
        "Starts finished": n_finished,
        "Starts matching kept loss": n_match,
        "Starts on a bound": n_on_bound,
        "Kept run on a bound": best["on_bound"],
        "abs(C) > 1": abs(best["C"]) > 1,
    }
    out = pd.DataFrame([row])
    xlsx = os.path.join(OUTPUT_FOLDER, f"{CRASH_NAME.replace(' ', '_')}_{tag}_Filimonov.xlsx")
    out.to_excel(xlsx, index=False)
    print(out.T.to_string())
    print(f"Saved: {xlsx}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], color="C0", label="Actual Sensex")
    ax.plot(df["Date"], np.exp(yhat), color="C3", ls="--", label="Filimonov fit")
    ax.axvline(tc_date, color="k", ls=":", label="Estimated tc")
    ax.set_title(f"{CRASH_NAME} — Filimonov split ({tag})")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(OUTPUT_FOLDER, f"{CRASH_NAME.replace(' ', '_')}_{tag}_Filimonov.png")
    fig.savefig(png, dpi=200)
    plt.close()
    print(f"Saved: {png}")

if __name__ == "__main__":
    fit_filimonov()