import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from construction_delay_model import generate_dataset, preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Delay Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-left: 4px solid #534AB7;
        margin-bottom: 0.5rem;
    }
    .risk-high   { border-left-color: #D85A30 !important; }
    .risk-medium { border-left-color: #EF9F27 !important; }
    .risk-low    { border-left-color: #1D9E75 !important; }
    .section-header {
        font-size: 1rem; font-weight: 600;
        color: #3C3489; margin-bottom: 0.3rem;
        border-bottom: 2px solid #EEEDFE;
        padding-bottom: 4px;
    }
    .stButton > button {
        background: #534AB7; color: white;
        border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #3C3489; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("best_model.pkl"):
        model = joblib.load("best_model.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        df = generate_dataset(500)
        df = preprocess(df)
        feature_cols = [c for c in df.columns if c != "delayed"]
        X, y = df[feature_cols], df["delayed"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        model.fit(X_train, y_train)
        joblib.dump(model, "best_model.pkl")
        joblib.dump(feature_cols, "feature_cols.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# ── Category encodings ─────────────────────────────────────────────────────────
project_type_map  = {"Residential": 2, "Commercial": 0, "Infrastructure": 1, "Industrial": 3}
contract_type_map = {"Lump Sum": 2, "Design-Build": 0, "Cost Plus": 1, "Unit Price": 3}
location_map      = {"Selangor": 4, "Kuala Lumpur": 1, "Johor": 0, "Penang": 3, "Sabah": 5, "Sarawak": 2}
funding_map       = {"Government": 0, "Private": 1, "PPP": 2}

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:2.8rem;margin-top:8px'>🏗️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("## Construction Project Delay Predictor")
    st.markdown("<span style='color:#888;font-size:0.9rem'>ML-powered delay risk assessment — Logistic Regression | Random Forest | XGBoost</span>", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar: input form ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Project Details")
    st.markdown("<div class='section-header'>Project Info</div>", unsafe_allow_html=True)

    project_type  = st.selectbox("Project type", list(project_type_map.keys()))
    contract_type = st.selectbox("Contract type", list(contract_type_map.keys()))
    location      = st.selectbox("Location", list(location_map.keys()))
    funding_type  = st.selectbox("Funding type", list(funding_map.keys()))

    st.markdown("<div class='section-header'>Schedule & Size</div>", unsafe_allow_html=True)
    planned_duration = st.slider("Planned duration (days)", 60, 1200, 365)
    project_size     = st.slider("Project size (m²)", 500, 50000, 5000, step=500)
    num_floors       = st.slider("Number of floors", 1, 40, 8)
    start_month      = st.selectbox("Start month", list(range(1, 13)),
                                    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                            "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])

    st.markdown("<div class='section-header'>Financial</div>", unsafe_allow_html=True)
    planned_budget      = st.number_input("Planned budget (RM)", 200000, 50000000, 3000000, step=100000)
    budget_variance_pct = st.slider("Budget variance (%)", -5.0, 40.0, 10.0, step=0.5)
    payment_delay_days  = st.slider("Payment delay (days)", 0, 90, 20)

    st.markdown("<div class='section-header'>Procurement & Team</div>", unsafe_allow_html=True)
    num_subcontractors    = st.slider("No. of subcontractors", 1, 20, 6)
    contractor_experience = st.slider("Contractor experience (years)", 1, 30, 10)
    approval_wait_days    = st.slider("Approval wait time (days)", 1, 120, 30)
    material_lead_days    = st.slider("Material lead time (days)", 5, 90, 25)
    decision_making_speed = st.slider("Decision-making speed (0=slow, 1=fast)", 0.0, 1.0, 0.5, step=0.05)

    st.markdown("<div class='section-header'>External Factors</div>", unsafe_allow_html=True)
    weather_risk_index  = st.slider("Weather risk index (0–1)", 0.0, 1.0, 0.4, step=0.05)
    labour_supply_score = st.slider("Labour supply score (0–1)", 0.0, 1.0, 0.6, step=0.05)
    inspection_freq     = st.slider("Inspection frequency", 1, 20, 8)
    public_holidays     = st.slider("Public holidays during project", 0, 15, 5)
    past_delay_rate     = st.slider("Contractor past delay rate (0–1)", 0.0, 1.0, 0.3, step=0.05)

    predict_btn = st.button("🔍 Predict Delay Risk")

# ── Derived features ───────────────────────────────────────────────────────────
def build_input():
    season_flag = 1 if start_month in [11, 12, 1, 2] else 0
    return {
        "project_type":          project_type_map[project_type],
        "contract_type":         contract_type_map[contract_type],
        "location":              location_map[location],
        "funding_type":          funding_map[funding_type],
        "planned_duration_days": planned_duration,
        "project_size_m2":       project_size,
        "num_floors":            num_floors,
        "num_subcontractors":    num_subcontractors,
        "contractor_experience": contractor_experience,
        "planned_budget_RM":     planned_budget,
        "budget_variance_pct":   budget_variance_pct,
        "payment_delay_days":    payment_delay_days,
        "approval_wait_days":    approval_wait_days,
        "material_lead_days":    material_lead_days,
        "weather_risk_index":    weather_risk_index,
        "labour_supply_score":   labour_supply_score,
        "decision_making_speed": decision_making_speed,
        "inspection_frequency":  inspection_freq,
        "start_month":           start_month,
        "public_holidays_count": public_holidays,
        "past_delay_rate":       past_delay_rate,
        "budget_per_day":        planned_budget / planned_duration,
        "size_duration_index":   project_size * planned_duration / 1e6,
        "subcontractor_ratio":   num_subcontractors / max(planned_duration, 1),
        "season_risk_flag":      season_flag,
        "complexity_score":      num_floors * 0.3 + num_subcontractors * 0.4 + project_size / 10000 * 0.3,
    }

# ── Always show live risk (auto-predict) ──────────────────────────────────────
input_data = build_input()
X_input = pd.DataFrame([input_data])[feature_cols]
prob  = model.predict_proba(X_input)[0][1]
label = int(prob >= 0.5)
risk_pct = round(prob * 100, 1)

if risk_pct >= 70:
    risk_level = "HIGH RISK"; risk_color = "#D85A30"; risk_css = "risk-high"
elif risk_pct >= 40:
    risk_level = "MEDIUM RISK"; risk_color = "#EF9F27"; risk_css = "risk-medium"
else:
    risk_level = "LOW RISK"; risk_color = "#1D9E75"; risk_css = "risk-low"

# ── Main area ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.markdown("#### Delay Risk Assessment")

    # Gauge chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 38, "color": risk_color}},
        delta={"reference": 50, "increasing": {"color": "#D85A30"}, "decreasing": {"color": "#1D9E75"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "#E1F5EE"},
                {"range": [40, 70],  "color": "#FAEEDA"},
                {"range": [70, 100], "color": "#FAECE7"},
            ],
            "threshold": {
                "line": {"color": risk_color, "width": 3},
                "thickness": 0.75,
                "value": risk_pct,
            },
        },
        title={"text": f"<b>{risk_level}</b>", "font": {"size": 14, "color": risk_color}},
    ))
    gauge.update_layout(
        height=260, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "sans-serif"},
    )
    st.plotly_chart(gauge, use_container_width=True)

    # Verdict card
    verdict = "⚠️ This project is likely to be DELAYED." if label else "✅ This project is likely to finish ON TIME."
    st.markdown(f"""
    <div class='metric-card {risk_css}'>
        <div style='font-size:1.05rem;font-weight:600;color:{risk_color}'>{verdict}</div>
        <div style='font-size:0.85rem;color:#666;margin-top:4px'>
            Delay risk: <b>{risk_pct}%</b> &nbsp;|&nbsp; Prediction: <b>{'Delayed' if label else 'Not Delayed'}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    st.markdown("#### Key Risk Indicators")
    def risk_badge(val, lo, hi):
        if val >= hi: return "🔴"
        elif val >= lo: return "🟡"
        return "🟢"

    indicators = [
        ("Budget variance",     f"{budget_variance_pct}%",     risk_badge(budget_variance_pct, 10, 25)),
        ("Payment delay",       f"{payment_delay_days} days",  risk_badge(payment_delay_days, 20, 50)),
        ("Approval wait",       f"{approval_wait_days} days",  risk_badge(approval_wait_days, 30, 70)),
        ("Weather risk",        f"{weather_risk_index:.2f}",   risk_badge(weather_risk_index, 0.4, 0.7)),
        ("Past delay rate",     f"{past_delay_rate:.2f}",      risk_badge(past_delay_rate, 0.3, 0.6)),
        ("Decision speed",      f"{decision_making_speed:.2f}",risk_badge(1-decision_making_speed, 0.4, 0.7)),
    ]
    for label_text, val, badge in indicators:
        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;font-size:0.88rem;border-bottom:1px solid #f0f0f0'>"
                    f"<span style='color:#555'>{label_text}</span>"
                    f"<span>{badge} <b>{val}</b></span></div>", unsafe_allow_html=True)

with col2:
    st.markdown("#### Feature Risk Contribution")

    # Risk factor breakdown bar chart
    risk_factors = {
        "Budget variance":      (budget_variance_pct / 40) * 0.20,
        "Payment delay":        (payment_delay_days / 90) * 0.15,
        "Approval wait":        (approval_wait_days / 120) * 0.15,
        "Weather risk":         weather_risk_index * 0.10,
        "Past delay rate":      past_delay_rate * 0.10,
        "Decision making":      (1 - decision_making_speed) * 0.10,
        "Material lead time":   (material_lead_days / 90) * 0.08,
        "Labour supply":        (1 - labour_supply_score) * 0.06,
        "Season risk":          (1 if start_month in [11,12,1,2] else 0) * 0.06,
    }
    rf_df = pd.DataFrame({
        "Factor": list(risk_factors.keys()),
        "Score":  [round(v * 100, 1) for v in risk_factors.values()],
    }).sort_values("Score", ascending=True)

    bar_colors = ["#D85A30" if s >= 8 else "#EF9F27" if s >= 4 else "#1D9E75"
                  for s in rf_df["Score"]]

    fig_bar = go.Figure(go.Bar(
        x=rf_df["Score"], y=rf_df["Factor"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}" for s in rf_df["Score"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=310, margin=dict(t=10, b=20, l=10, r=50),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Risk contribution (%)", gridcolor="#f0f0f0"),
        yaxis=dict(tickfont=dict(size=12)),
        font=dict(family="sans-serif", size=12),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Project summary table
    st.markdown("#### Project Summary")
    summary = {
        "Project type":       project_type,
        "Contract type":      contract_type,
        "Location":           location,
        "Funding":            funding_type,
        "Duration":           f"{planned_duration} days",
        "Budget":             f"RM {planned_budget:,.0f}",
        "Size":               f"{project_size:,} m²",
        "Floors":             num_floors,
        "Subcontractors":     num_subcontractors,
        "Complexity score":   f"{input_data['complexity_score']:.2f}",
    }
    summary_df = pd.DataFrame(summary.items(), columns=["Parameter", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True,
                 column_config={"Parameter": st.column_config.TextColumn(width="medium"),
                                "Value":     st.column_config.TextColumn(width="medium")})

# ── Historical dataset overview ────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Dataset Overview")

@st.cache_data
def get_overview_data():
    df = generate_dataset(500)
    df = preprocess(df)
    return df

df_overview = get_overview_data()

oc1, oc2, oc3, oc4 = st.columns(4)
with oc1:
    st.metric("Total projects", len(df_overview))
with oc2:
    st.metric("Delayed", f"{df_overview['delayed'].sum()} ({df_overview['delayed'].mean()*100:.0f}%)")
with oc3:
    st.metric("Avg duration", f"{df_overview['planned_duration_days'].mean():.0f} days")
with oc4:
    st.metric("Avg budget", f"RM {df_overview['planned_budget_RM'].mean():,.0f}")

dc1, dc2 = st.columns(2)
with dc1:
    delay_counts = df_overview["delayed"].value_counts().reset_index()
    delay_counts.columns = ["Status", "Count"]
    delay_counts["Status"] = delay_counts["Status"].map({1: "Delayed", 0: "Not Delayed"})
    fig_pie = px.pie(delay_counts, values="Count", names="Status",
                     color_discrete_map={"Delayed": "#D85A30", "Not Delayed": "#1D9E75"},
                     title="Delay distribution")
    fig_pie.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                          paper_bgcolor="white")
    st.plotly_chart(fig_pie, use_container_width=True)

with dc2:
    fig_hist = px.histogram(df_overview, x="budget_variance_pct", color="delayed",
                             color_discrete_map={1: "#D85A30", 0: "#1D9E75"},
                             labels={"delayed": "Delayed", "budget_variance_pct": "Budget variance (%)"},
                             title="Budget variance vs delay",
                             barmode="overlay", opacity=0.7)
    fig_hist.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                           paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;margin-top:1rem'>"
    "Construction Project Delay Predictor · UiTM · 2025397057@student.uitm.edu.my"
    "</div>", unsafe_allow_html=True
)
