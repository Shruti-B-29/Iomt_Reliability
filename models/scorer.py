import numpy as np
import pandas as pd
from scipy import stats

# ── Window size ───────────────────────────────────────────────
WINDOW_SIZE = 30

# ── Sub-component weights (from notebook) ────────────────
W_WITHIN   = [0.15, 0.15, 0.55, 0.15]   # Ca, Cb, Cc, Cd
W_TEMPORAL = [0.25, 0.25, 0.30, 0.20]   # Sm, Sj, Sf, Sd
W_FINAL    = [0.70, 0.15, 0.15]          # w1, w2, w3

# ── Saturation scales ─────────────────────────────────────────
REF_MAE_HR   = 20.0
REF_MAE_SPO2 = 2.0

# ── Rule-based reference (from your notebook output) ──────────
RB_REF = {
    'hr':   {'mean': 75.93, 'std': 7.93,  'window_var': 14.7053},
    'spo2': {'mean': 97.55, 'std': 1.656, 'window_var': 0.0695},
}

# ─────────────────────────────────────────────────────────────
def safe_corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    try:
        r, _ = stats.pearsonr(x, y)
        return 0.0 if np.isnan(r) else r
    except:
        return 0.0

# ─────────────────────────────────────────────────────────────
def S_within(w: pd.DataFrame) -> float:
    """Within-sensor consistency score."""
    rh = w['pulse'].dropna().values
    rs = w['spo2'].dropna().values
    dh = w['gen_pulse'].dropna().values
    ds = w['gen_spo2'].dropna().values

    if len(rh) < 5 or len(rs) < 5:
        return np.nan

    rb = RB_REF

    # Ca: Global mean alignment with rule-based reference
    Ca = (0.5 * np.exp(-0.5 * (abs(rh.mean() - rb['hr']['mean'])   / (rb['hr']['std']   + 1e-9)) ** 2)
        + 0.5 * np.exp(-0.5 * (abs(rs.mean() - rb['spo2']['mean']) / (rb['spo2']['std'] + 1e-9)) ** 2))

    # Cb: Within-window variance alignment (log-space penalty)
    hr_var   = np.var(rh) + 1e-9
    spo2_var = np.var(rs) + 1e-9
    Cb = (0.5 * np.exp(-abs(np.log(hr_var   / rb['hr']['window_var'])))
        + 0.5 * np.exp(-abs(np.log(spo2_var / rb['spo2']['window_var']))))

    # Cc: Quantisation integrity — MAE between real and DL (KEY discriminator)
    n_h = min(len(rh), len(dh))
    n_s = min(len(rs), len(ds))
    if n_h >= 5 and n_s >= 5:
        mae_hr   = np.mean(np.abs(rh[:n_h] - dh[:n_h]))
        mae_spo2 = np.mean(np.abs(rs[:n_s] - ds[:n_s]))
        Cc = (0.5 * np.tanh(mae_hr   / REF_MAE_HR)
            + 0.5 * np.tanh(mae_spo2 / REF_MAE_SPO2))
    else:
        Cc = 0.0

    # Cd: Intra-window drift (mean shift first vs second half)
    mid = len(rh) // 2
    drift_hr   = abs(rh[mid:].mean() - rh[:mid].mean()) / (np.std(rh)   + 1e-9)
    drift_spo2 = abs(rs[mid:].mean() - rs[:mid].mean()) / (np.std(rs) + 1e-9)
    Cd = 1.0 / (1.0 + 0.5 * (drift_hr + drift_spo2))

    return float(W_WITHIN[0]*Ca + W_WITHIN[1]*Cb + W_WITHIN[2]*Cc + W_WITHIN[3]*Cd)

# ─────────────────────────────────────────────────────────────
def S_temporal(w: pd.DataFrame) -> float:
    """Temporal stability score."""
    rh = w['pulse'].dropna().values
    rs = w['spo2'].dropna().values

    if len(rh) < 5 or len(rs) < 5:
        return np.nan

    # Sm: Mean stability (how far from rule-based mean)
    Sm = (0.5 * np.exp(-0.5 * ((rh.mean() - RB_REF['hr']['mean'])   / RB_REF['hr']['std'])   ** 2)
        + 0.5 * np.exp(-0.5 * ((rs.mean() - RB_REF['spo2']['mean']) / RB_REF['spo2']['std']) ** 2))

    # Sj: Jitter (point-to-point variation)
    jitter_hr   = np.mean(np.abs(np.diff(rh))) / (RB_REF['hr']['std']   + 1e-9)
    jitter_spo2 = np.mean(np.abs(np.diff(rs))) / (RB_REF['spo2']['std'] + 1e-9)
    Sj = np.exp(-0.5 * (jitter_hr + jitter_spo2))

    # Sf: Freeze detection (fraction of repeated values)
    freeze_hr   = np.mean(np.diff(rh) == 0)
    freeze_spo2 = np.mean(np.diff(rs) == 0)
    Sf = 1.0 - 0.5 * (freeze_hr + freeze_spo2)

    # Sd: Drift (normalised range)
    drift_hr   = (rh.max() - rh.min()) / (RB_REF['hr']['std']   * 6 + 1e-9)
    drift_spo2 = (rs.max() - rs.min()) / (RB_REF['spo2']['std'] * 6 + 1e-9)
    Sd = np.exp(-0.3 * (drift_hr + drift_spo2))

    return float(W_TEMPORAL[0]*Sm + W_TEMPORAL[1]*Sj + W_TEMPORAL[2]*Sf + W_TEMPORAL[3]*Sd)

# ─────────────────────────────────────────────────────────────
def S_cross(w: pd.DataFrame) -> float:
    """Cross-sensor consistency score."""
    rh = w['pulse'].dropna().values
    rs = w['spo2'].dropna().values
    dh = w['gen_pulse'].dropna().values
    ds = w['gen_spo2'].dropna().values

    n = min(len(rh), len(rs))
    if n < 5:
        return np.nan

    # Real cross-correlation
    r_real = abs(safe_corr(rh[:n], rs[:n]))

    # DL cross-correlation
    n_dl = min(len(dh), len(ds))
    r_dl = abs(safe_corr(dh[:n_dl], ds[:n_dl])) if n_dl >= 5 else 0.0

    # Attack penalty: attacks create spurious high DL correlation
    attack_penalty = 1.0 - min(r_dl, 1.0)

    # Base score: low real cross-correlation = good (HR and SpO2 shouldn't correlate strongly)
    base = np.exp(-2.0 * r_real)

    return float(0.6 * base + 0.4 * attack_penalty)

# ─────────────────────────────────────────────────────────────
def score_window(w: pd.DataFrame) -> dict:
    """Score a single window of WINDOW_SIZE rows."""
    sw = S_within(w)
    st = S_temporal(w)
    sc = S_cross(w)

    if any(np.isnan(v) for v in [sw, st, sc]):
        return {'error': 'Insufficient data in window'}

    R = (W_FINAL[0]*sw + W_FINAL[1]*st + W_FINAL[2]*sc) * 100

    return {
        'S_within':   round(sw, 4),
        'S_temporal': round(st, 4),
        'S_cross':    round(sc, 4),
        'R':          round(R,  2),
        'status':     classify_score(R),
    }

# ─────────────────────────────────────────────────────────────
def score_condition(df: pd.DataFrame) -> dict:
    """Score all windows in a condition dataframe."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    n_windows = len(df) // WINDOW_SIZE
    results = []

    for i in range(n_windows):
        w = df.iloc[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]
        s = score_window(w)
        if 'error' not in s:
            results.append(s)

    if not results:
        return {'error': 'No valid windows'}

    R_vals = [r['R'] for r in results]
    sw_vals = [r['S_within']   for r in results]
    st_vals = [r['S_temporal'] for r in results]
    sc_vals = [r['S_cross']    for r in results]

    return {
        'n_windows':   len(results),
        'R_mean':      round(float(np.mean(R_vals)),  2),
        'R_std':       round(float(np.std(R_vals)),   2),
        'R_min':       round(float(np.min(R_vals)),   2),
        'R_max':       round(float(np.max(R_vals)),   2),
        'S_within':    round(float(np.mean(sw_vals)), 4),
        'S_temporal':  round(float(np.mean(st_vals)), 4),
        'S_cross':     round(float(np.mean(sc_vals)), 4),
        'status':      classify_score(np.mean(R_vals)),
        'per_window':  results,
    }

# ─────────────────────────────────────────────────────────────
def classify_score(R: float) -> str:
    if R >= 55:   return 'NORMAL'
    elif R >= 45: return 'MARGINAL'
    elif R >= 35: return 'ANOMALOUS'
    else:         return 'CRITICAL'