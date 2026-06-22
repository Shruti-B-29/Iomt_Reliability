import pandas as pd
import sys
sys.path.append('..')
from models.scorer import score_condition

DATA_PATH = 'data/IoMT Sensor Data/DL Readings/Synthetic and Real/'

files = {
    'Normal':     'pulse_oximeter_normal_normal.csv',
    'Noise':      'pulse_oximeter_device_anomaly_noise.csv',
    'Freeze':     'pulse_oximeter_device_anomaly_freeze.csv',
    'Bradycardia':'pulse_oximeter_patient_anomaly_bradycardia.csv',
    'Replay':     'pulse_oximeter_attack_replay_attack.csv',
}

print(f"{'Condition':<16} {'R_mean':>8} {'R_std':>7} {'S_within':>10} {'Status'}")
print('─' * 60)
for name, fname in files.items():
    df = pd.read_csv(DATA_PATH + fname)
    result = score_condition(df)
    print(f"{name:<16} {result['R_mean']:>8.2f} {result['R_std']:>7.2f} "
          f"{result['S_within']:>10.4f}  {result['status']}")