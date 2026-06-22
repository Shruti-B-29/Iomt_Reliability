import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.scorer import score_condition

st.set_page_config(page_title="IoMT Reliability Dashboard", page_icon="🩺", layout="wide")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')

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

@st.cache_data(ttl=3600)
def fetch_all_scores():
    results = {}
    for name, fname in ALL_CONDITIONS.items():
        fpath = os.path.join(DATA_PATH, fname)
        df = pd.read_csv(fpath)
        r = score_condition(df)
        results[name] = {
            "R_mean": r['R_mean'], "R_std": r['R_std'],
            "S_within": r['S_within'], "S_temporal": r['S_temporal'], "S_cross": r['S_cross'],
            "status": r['status'], "n_windows": r['n_windows'],
        }
    ranked = dict(sorted(results.items(), key=lambda x: x[1]['R_mean'], reverse=True))
    return {
        "summary": ranked,
        "normal_R": results['normal']['R_mean'],
        "lowest_R": min(v['R_mean'] for v in ranked.values()),
        "all_normal_highest": all(
            results['normal']['R_mean'] > v['R_mean']
            for k, v in ranked.items() if k != 'normal'
        )
    }

@st.cache_data(ttl=3600)
def fetch_condition(name):
    fpath = os.path.join(DATA_PATH, ALL_CONDITIONS[name])
    df = pd.read_csv(fpath)
    return score_condition(df)

st.title("🩺 IoMT Pulse Oximeter Reliability Dashboard")
st.markdown("**Statistical Reliability Scoring · R = 0.70·S_within + 0.15·S_temporal + 0.15·S_cross**")
st.markdown("---")

data = fetch_all_scores()

summary = data['summary']
df = pd.DataFrame(summary).T.reset_index().rename(columns={'index': 'condition'})
df['category'] = df['condition'].apply(lambda c:
    'Normal' if c == 'normal' else
    'Device Anomaly' if c in ['noise','freeze','drift','packet_loss'] else
    'Patient Anomaly' if c in ['bradycardia','tachycardia','hypoxemia','rapid_drop'] else
    'Attack'
)

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Normal R Score", f"{data['normal_R']:.2f}")
k2.metric("Lowest R Score", f"{data['lowest_R']:.2f}")
k3.metric("Separation", f"{data['normal_R'] - data['lowest_R']:.2f}")
k4.metric("Discrimination", "✅ PASS" if data['all_normal_highest'] else "❌ FAIL")

st.markdown("---")

# ── Ranked bar chart ──────────────────────────────────────────
st.subheader("📊 Reliability Score by Condition")
df_sorted = df.sort_values('R_mean', ascending=True)

color_map = {'Normal': '#2ECC71', 'Device Anomaly': '#E67E22',
             'Patient Anomaly': '#E74C3C', 'Attack': '#8E44AD'}

fig = px.bar(df_sorted, x='R_mean', y='condition', color='category',
             orientation='h', color_discrete_map=color_map,
             error_x='R_std',
             labels={'R_mean': 'Reliability Score R (%)', 'condition': ''},
             title='')
fig.add_vline(x=55, line_dash="dash", line_color="green", annotation_text="NORMAL threshold")
fig.add_vline(x=45, line_dash="dash", line_color="orange", annotation_text="MARGINAL threshold")
fig.add_vline(x=35, line_dash="dash", line_color="red", annotation_text="CRITICAL threshold")
fig.update_layout(height=550, legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Sub-score breakdown ───────────────────────────────────────
st.subheader("Sub-Score Breakdown")
col1, col2 = st.columns([2,1])

with col1:
    fig2 = go.Figure()
    for comp, color in [('S_within','#3498DB'), ('S_temporal','#F1C40F'), ('S_cross','#E74C3C')]:
        fig2.add_trace(go.Bar(name=comp, x=df_sorted['condition'], y=df_sorted[comp], marker_color=color))
    fig2.update_layout(barmode='group', height=420, xaxis_tickangle=-45,
                        title='S_within / S_temporal / S_cross per Condition')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("**Component Weights**")
    st.markdown("""
    | Component | Weight | Role |
    |---|---|---|
    | S_within | 0.70 | DL vs Real MAE (key discriminator) |
    | S_temporal | 0.15 | Jitter, freeze, drift |
    | S_cross | 0.15 | Cross-sensor correlation |
    """)
    st.info("**S_within dominates** — the quantisation integrity term (MAE between real and DL-generated readings) is the strongest signal separating Normal from anomalies.")

st.markdown("---")

# ── Detailed table ────────────────────────────────────────────
st.subheader("Full Results Table")
display_df = df_sorted[['condition','category','R_mean','R_std','S_within','S_temporal','S_cross','status','n_windows']].sort_values('R_mean', ascending=False)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Single condition deep-dive ─────────────────────────────────
st.subheader("Per-Window Deep Dive")
selected = st.selectbox("Select a condition to inspect window-by-window scores", df['condition'].tolist())

detail = fetch_condition(selected)
windows_df = pd.DataFrame(detail['per_window'])
windows_df['window_idx'] = range(len(windows_df))

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=windows_df['window_idx'], y=windows_df['R'],
                          mode='lines+markers', line=dict(color='#3498DB')))
fig3.add_hline(y=55, line_dash="dash", line_color="green")
fig3.add_hline(y=45, line_dash="dash", line_color="orange")
fig3.add_hline(y=35, line_dash="dash", line_color="red")
fig3.update_layout(title=f"R Score per Window — {selected}", height=350,
                    xaxis_title="Window Index", yaxis_title="R Score")
st.plotly_chart(fig3, use_container_width=True)

st.caption("Built with FastAPI (local) · Streamlit · Plotly | Backend: composite statistical reliability scoring")