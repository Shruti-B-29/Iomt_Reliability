from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.scorer import score_window, score_condition, WINDOW_SIZE, classify_score

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="IoMT Reliability Scoring API",
    description="""
    Statistical reliability scoring for IoMT pulse oximeter data.
    Detects normal operation vs 12 anomaly/attack conditions using
    a composite score R = 0.70·S_within + 0.15·S_temporal + 0.15·S_cross.
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data path ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')

print(f"DATA_PATH: {DATA_PATH}")
print(f"Files found: {os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else 'PATH NOT FOUND'}")

ALL_CONDITIONS = {
    'normal':               'pulse_oximeter_normal_normal.csv',
    'noise':                'pulse_oximeter_device_anomaly_noise.csv',
    'freeze':               'pulse_oximeter_device_anomaly_freeze.csv',
    'drift':                'pulse_oximeter_device_anomaly_drift.csv',
    'packet_loss':          'pulse_oximeter_device_anomaly_packet_loss.csv',
    'bradycardia':          'pulse_oximeter_patient_anomaly_bradycardia.csv',
    'tachycardia':          'pulse_oximeter_patient_anomaly_tachycardia.csv',
    'hypoxemia':            'pulse_oximeter_patient_anomaly_hypoxemia.csv',
    'rapid_drop':           'pulse_oximeter_patient_anomaly_rapid_drop.csv',
    'replay_attack':        'pulse_oximeter_attack_replay_attack.csv',
    'false_data_injection': 'pulse_oximeter_attack_false_data_injection.csv',
    'selective_forwarding': 'pulse_oximeter_attack_selective_forwarding.csv',
    'temporal_attack':      'pulse_oximeter_attack_temporal_attack.csv',
}

# ── Request models ────────────────────────────────────────────
class WindowRequest(BaseModel):
    pulse:     list[float]
    spo2:      list[float]
    gen_pulse: list[Optional[float]]
    gen_spo2:  list[Optional[float]]

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "IoMT Reliability Scoring API",
        "version": "1.0.0",
        "endpoints": ["/score/window", "/score/condition/{name}", "/score/all", "/conditions"]
    }

@app.get("/conditions")
def list_conditions():
    """List all available conditions."""
    return {
        "conditions": list(ALL_CONDITIONS.keys()),
        "total": len(ALL_CONDITIONS),
        "categories": {
            "normal":           ["normal"],
            "device_anomaly":   ["noise", "freeze", "drift", "packet_loss"],
            "patient_anomaly":  ["bradycardia", "tachycardia", "hypoxemia", "rapid_drop"],
            "attack":           ["replay_attack", "false_data_injection",
                                 "selective_forwarding", "temporal_attack"]
        }
    }

@app.post("/score/window")
def score_single_window(req: WindowRequest):
    """
    Score a single window of pulse oximeter readings.
    Send exactly 30 readings per sensor for best results.
    Returns R score, sub-scores, and status classification.
    """
    if len(req.pulse) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 readings required")

    df = pd.DataFrame({
        'pulse':     req.pulse,
        'spo2':      req.spo2,
        'gen_pulse': req.gen_pulse,
        'gen_spo2':  req.gen_spo2,
        'timestamp': range(len(req.pulse))
    })

    result = score_window(df)
    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])

    return {
        "window_size":  len(req.pulse),
        "R":            result['R'],
        "S_within":     result['S_within'],
        "S_temporal":   result['S_temporal'],
        "S_cross":      result['S_cross'],
        "status":       result['status'],
        "interpretation": {
            "NORMAL":    "Device operating within expected parameters",
            "MARGINAL":  "Minor deviation detected — monitor closely",
            "ANOMALOUS": "Significant anomaly detected — investigate",
            "CRITICAL":  "Critical failure — immediate action required"
        }[result['status']]
    }

@app.get("/score/condition/{condition_name}")
def score_named_condition(condition_name: str):
    """
    Score all windows for a named condition from the dataset.
    Returns mean R, std, per-window breakdown, and status.
    """
    if condition_name not in ALL_CONDITIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Condition '{condition_name}' not found. "
                   f"Available: {list(ALL_CONDITIONS.keys())}"
        )

    fpath = os.path.join(DATA_PATH, ALL_CONDITIONS[condition_name])
    if not os.path.exists(fpath):
        raise HTTPException(status_code=500, detail=f"Data file not found: {fpath}")

    df = pd.read_csv(fpath)
    result = score_condition(df)

    return {
        "condition":   condition_name,
        "n_windows":   result['n_windows'],
        "R_mean":      result['R_mean'],
        "R_std":       result['R_std'],
        "R_min":       result['R_min'],
        "R_max":       result['R_max'],
        "S_within":    result['S_within'],
        "S_temporal":  result['S_temporal'],
        "S_cross":     result['S_cross'],
        "status":      result['status'],
        "per_window":  result['per_window']
    }

@app.get("/score/all")
def score_all_conditions():
    """
    Score all 13 conditions and return a ranked summary.
    This is the main endpoint for dashboard population.
    """
    results = {}
    for name, fname in ALL_CONDITIONS.items():
        fpath = os.path.join(DATA_PATH, fname)
        if not os.path.exists(fpath):
            results[name] = {"error": f"File not found: {fname}"}
            continue
        df = pd.read_csv(fpath)
        r  = score_condition(df)
        results[name] = {
            "R_mean":     r['R_mean'],
            "R_std":      r['R_std'],
            "S_within":   r['S_within'],
            "S_temporal": r['S_temporal'],
            "S_cross":    r['S_cross'],
            "status":     r['status'],
            "n_windows":  r['n_windows'],
        }

    # Sort by R_mean descending
    ranked = dict(sorted(
        {k: v for k, v in results.items() if 'error' not in v}.items(),
        key=lambda x: x[1]['R_mean'],
        reverse=True
    ))

    return {
        "summary": ranked,
        "normal_R": results.get('normal', {}).get('R_mean'),
        "lowest_R": min((v['R_mean'] for v in ranked.values()), default=None),
        "all_normal_highest": (
    all(results['normal']['R_mean'] > v['R_mean']
        for k, v in ranked.items() if k != 'normal')
    if 'normal' in results and 'error' not in results.get('normal', {})
    else False
)
    }

@app.post("/score/upload")
async def score_uploaded_file(file: UploadFile = File(...)):
    """
    Upload a CSV file and score all its windows.
    Expected columns: pulse, spo2, gen_pulse, gen_spo2, timestamp
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    required = ['pulse', 'spo2', 'gen_pulse', 'gen_spo2']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422,
            detail=f"Missing columns: {missing}. Required: {required}")

    result = score_condition(df)
    return {
        "filename":  file.filename,
        "rows":      len(df),
        "n_windows": result['n_windows'],
        "R_mean":    result['R_mean'],
        "R_std":     result['R_std'],
        "status":    result['status'],
        "per_window": result['per_window']
    }