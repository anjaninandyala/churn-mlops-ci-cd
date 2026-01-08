import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import base64
import requests

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000/predict"   # default for local run
)


# -----------------------
# Paths
# -----------------------
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
COLUMNS_PATH = "models/columns.pkl"
RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
# ========================
# MODEL PERFORMANCE PATHS
# ========================
LEADERBOARD_PATH = "models/leaderboard.csv"
METRICS_PATH = "models/metrics.json"
REPORTS_DIR = "reports"


# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Churn BI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------
# Light CSS – BI + slight neon
# -----------------------
st.markdown(
    """
<style>
    body { background-color: #020617; color: #e5e7eb; }
    .main {
        background: radial-gradient(circle at top, #020617 0, #020617 55%);
        color: #e5e7eb;
    }
    h1, h2, h3, h4 { color: #e5e7eb !important; }

    /* Metric cards */
    .metric-card {
        padding: 16px 18px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.98);
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .metric-label { color:#9ca3af; font-size:0.8rem; }
    .metric-value { color:#e5e7eb; font-size:1.6rem; font-weight:600; }

    /* --- FIX UNDERLINE BELOW TABS COMPLETELY --- */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

.stTabs [data-baseweb="tab"]::after {
    border-bottom: none !important;
}

/* --- IMPROVED TAB STYLE --- */
.stTabs [data-baseweb="tab"] {
    border-radius: 999px !important;
    padding: 6px 18px !important;
    background-color: rgba(15,23,42,0.95) !important;
    color: #9ca3af !important;
    border: 1px solid transparent !important;
    margin-right: 6px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#22d3ee,#4ade80) !important;
    color: #020617 !important;
    font-weight: 700 !important;
}


    /* FIX 2: Improve form label visibility */
    label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* FIX 3: Dark sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Styled multiselect tags */
    .css-1n76uvr, .css-1n76uvr * {
        background-color: #1e293b !important;
        color: #e5e7eb !important;
        border: 1px solid #334155 !important;
    }

    /* Slider neon gradient */
    .stSlider > div > div > div:nth-child(2) > div {
        background: linear-gradient(90deg, #22d3ee, #4ade80) !important;
    }
.risk-bar-container {
    width: 100%;
    background: #0f172a;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.3);
}

.risk-bar {
    width: 100%;
    height: 22px;
    background: linear-gradient(to right,
        #22c55e 0%, #22c55e 33%,     /* Low */
        #facc15 33%, #facc15 66%,    /* Medium */
        #ef4444 66%, #ef4444 100%    /* High */
    );
    border-radius: 10px;
    position: relative;
}

.risk-indicator {
    position: absolute;
    top: -6px;
    width: 12px;
    height: 34px;
    background: white;
    border-radius: 6px;
    border: 2px solid black;
    transition: left 0.6s ease;
}

.risk-label {
    margin-top: 8px;
    font-size: 0.9rem;
    color: #94a3b8;
}

.risk-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.pill-low { background: rgba(34,197,94,0.2); color:#22c55e; }
.pill-med { background: rgba(234,179,8,0.2); color:#facc15; }
.pill-high { background: rgba(239,68,68,0.2); color:#ef4444; }


/* Glass container */
.glass-table {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Rank badges */
.rank-badge {
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 10px;
    color: #fff;
}
.rank-1 { background: linear-gradient(135deg,#facc15,#fde047); }
.rank-2 { background: linear-gradient(135deg,#9ca3af,#e5e7eb); }
.rank-3 { background: linear-gradient(135deg,#d97706,#fbbf24); }

/* Sparkline column */
.spark-cell {
    width: 120px;
}

/* Hover effect */
.table-hover tbody tr:hover {
    background-color: rgba(255,255,255,0.06) !important;
    cursor: pointer;
}

/* Export button */
.export-btn {
    background: linear-gradient(135deg,#22d3ee,#4ade80);
    padding: 8px 14px;
    border-radius: 10px;
    color: #000;
    font-weight: 700;
    margin-bottom: 10px;
    display: inline-block;
}

/* --- Stronger highlight rows --- */
.row-top-1 { background: rgba(255, 75, 75, 0.35) !important; }
.row-top-2 { background: rgba(255, 110, 110, 0.30) !important; }
.row-top-3 { background: rgba(255, 150, 150, 0.25) !important; }

/* Improve text contrast on highlighted rows */
.row-top-1 td, 
.row-top-2 td, 
.row-top-3 td {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Badges */
.badge-low {
    background: rgba(34,197,94,0.2);
    padding: 6px 12px;
    border-radius: 14px;
    color:#22c55e;
    font-weight:600;
}
.badge-med {
    background: rgba(234,179,8,0.2);
    padding: 6px 12px;
    border-radius: 14px;
    color:#eab308;
    font-weight:600;
}
.badge-high {
    background: rgba(239,68,68,0.2);
    padding: 6px 12px;
    border-radius: 14px;
    color:#ef4444;
    font-weight:600;
}
.profile-card {
    background: rgba(15, 23, 42, 0.7);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.25);
    margin-bottom: 12px;
}
.profile-card h3 {
    color: #fff;
    margin-bottom: 12px;
}
.profile-card p {
    color: #e2e8f0;
    font-size: 1rem;
}




</style>
""",
    unsafe_allow_html=True,
)
# -----------------------
# Modern Glass + Neon Plotly Theme
# -----------------------
import plotly.io as pio

pio.templates["glass_neon"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", family="Inter"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "glass_neon"




# -----------------------
# Load artifacts
# -----------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Run training pipeline.")
        return None, None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, encoders, columns


@st.cache_data
def load_raw_data():
    if not os.path.exists(RAW_DATA_PATH):
        return None
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_processed_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    return pd.read_csv(PROCESSED_DATA_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


model, scaler, encoders, columns = load_artifacts()
metrics = load_metrics()
raw_df = load_raw_data()
proc_df = load_processed_data()

# -----------------------
# Build scored dataset for BI
# -----------------------
@st.cache_data
def build_scored_df(raw_df, proc_df):
    if raw_df is None or proc_df is None or model is None:
        return None

    df_proc = proc_df.copy()
    X = df_proc.drop("Churn", axis=1)
    probs = model.predict_proba(X)[:, 1]

    df_view = raw_df.copy()
    df_view["churn_prob"] = probs
    df_view["churn_label"] = (df_view["Churn"] == "Yes").astype(int)

    def risk_bucket(p):
        if p < 0.33:
            return "Low"
        elif p < 0.66:
            return "Medium"
        else:
            return "High"

    df_view["risk_level"] = df_view["churn_prob"].apply(risk_bucket)
    return df_view


scored_df = build_scored_df(raw_df, proc_df)

# -----------------------
# Helper: preprocess single row for prediction
# -----------------------
def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df = df.reindex(columns=columns, fill_value=0)
    return df
def predict_via_backend(input_df):
    payload = input_df.to_dict(orient="records")[0]

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None



# ============================================================
# SIDEBAR – filters (BI style)
# ============================================================
st.sidebar.title("Filters")

if scored_df is not None:
    contracts = scored_df["Contract"].dropna().unique().tolist()
    contract_filter = st.sidebar.multiselect(
        "Contract Type", options=contracts, default=contracts
    )

    services = scored_df["InternetService"].dropna().unique().tolist()
    internet_filter = st.sidebar.multiselect(
        "Internet Service", options=services, default=services
    )

    min_tenure, max_tenure = int(scored_df["tenure"].min()), int(
        scored_df["tenure"].max()
    )
    tenure_range = st.sidebar.slider(
        "Tenure Range (months)", min_tenure, max_tenure, (min_tenure, max_tenure)
    )

    risk_levels = ["Low", "Medium", "High"]
    risk_filter = st.sidebar.multiselect(
        "Risk Level", options=risk_levels, default=risk_levels
    )
else:
    contract_filter = internet_filter = risk_filter = []
    tenure_range = (0, 72)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df_f = df.copy()
    if contract_filter:
        df_f = df_f[df_f["Contract"].isin(contract_filter)]
    if internet_filter:
        df_f = df_f[df_f["InternetService"].isin(internet_filter)]
    if risk_filter:
        df_f = df_f[df_f["risk_level"].isin(risk_filter)]
    df_f = df_f[(df_f["tenure"] >= tenure_range[0]) & (df_f["tenure"] <= tenure_range[1])]
    return df_f


filtered_df = apply_filters(scored_df) if scored_df is not None else None

# ============================================================
# HEADER
# ============================================================
st.title("Customer Churn Prediction System")
st.caption(
    "Business Intelligence view over churn data with interactive filters and real-time churn prediction."
)

# Tabs
tab_overview, tab_segments, tab_predict, tab_performance = st.tabs(
    ["Overview", "Segment Analysis", "Predict Churn", "Model Performance"]
)

# ============================================================
# TAB 1 – OVERVIEW
# ============================================================
with tab_overview:
    st.subheader("Key Metrics (after filters)")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        total_customers = len(filtered_df)
        churn_rate = (filtered_df["Churn"] == "Yes").mean()
        at_risk_revenue = filtered_df[filtered_df["churn_prob"] > 0.5][
            "MonthlyCharges"
        ].sum()
        high_risk_count = (filtered_df["risk_level"] == "High").sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                    <div class="metric-sub">Filtered population</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Churn Rate</div>
                    <div class="metric-value">{churn_rate*100:.1f}%</div>
                    <div class="metric-sub">% of customers who churned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">At-Risk Monthly Revenue</div>
                    <div class="metric-value">${at_risk_revenue:,.1f}</div>
                    <div class="metric-sub">From customers with churn_prob &gt; 0.5</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">High-Risk Customers</div>
                    <div class="metric-value">{high_risk_count}</div>
                    <div class="metric-sub">Risk level = High</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Charts row: tenure trend + risk distribution
        c_trend, c_risk = st.columns([0.6, 0.4])

        with c_trend:
            st.markdown("**Churn by Tenure Bucket**")

            trend_df = filtered_df.copy()
            trend_df["tenure_bucket"] = pd.cut(
                trend_df["tenure"],
                bins=[0, 6, 12, 24, 36, 48, 60, 72],
                include_lowest=True,
            ).astype(str)

            trend_chart = (
                trend_df.groupby(["tenure_bucket", "Churn"])["customerID"]
                .count()
                .reset_index()
                .rename(columns={"customerID": "Count"})
            )

            fig_trend = px.line(
                trend_chart,
                x="tenure_bucket",
                y="Count",
                color="Churn",
                markers=True,
                template="glass_neon"
,
            )
            fig_trend.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title_text=""
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with c_risk:
            st.markdown("**Risk Distribution**")
            risk_counts = (
                filtered_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"])
            )
            fig_donut = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.6,
                template="glass_neon",
                color=risk_counts.index,
                color_discrete_map={
                    "Low": "#22c55e",
                    "Medium": "#eab308",
                    "High": "#f97316",
                },
            )
            fig_donut.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("")

        # Churn by Contract
        st.markdown("**Churn Rate by Contract Type**")
        contract_df = (
            filtered_df.groupby("Contract")["churn_label"]
            .agg(["mean", "count"])
            .reset_index()
        )
        contract_df["ChurnRate(%)"] = (contract_df["mean"] * 100).round(1)
        fig_contract = px.bar(
            contract_df,
            x="Contract",
            y="ChurnRate(%)",
            text="ChurnRate(%)",
            template="glass_neon",
        )
        fig_contract.update_traces(textposition="outside")
        fig_contract.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Churn Rate (%)"
        )
        st.plotly_chart(fig_contract, use_container_width=True)

# ---------------- RETENTION RECOMMENDATION LOGIC ----------------
def retention_recommendation(row, short=True):
    recs = []

    if row["Contract"] == "Month-to-month":
        recs.append("Offer discount on long-term contract")

    if row["MonthlyCharges"] > 80:
        recs.append("Provide billing discount or plan optimization")

    if row["InternetService"] == "Fiber optic":
        recs.append("Check service quality / offer support")

    if row["tenure"] < 12:
        recs.append("Provide loyalty or welcome benefits")

    if not recs:
        recs.append("Maintain current engagement strategy")
    
    if short:
        return " | ".join(recs[:2]) 
    else:
        return recs





# ============================================================
# TAB 2 – SEGMENT ANALYSIS & FEATURE IMPORTANCE
# ============================================================
with tab_segments:

    st.subheader("Feature Importance (Model View)")

    if model is None or columns is None:
        st.warning("Model or columns not found.")
    else:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(12)

            fig_fi = px.bar(
                fi,
                x="importance",
                y="feature",
                orientation="h",
                template="glass_neon",
            )
            fig_fi.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Current model does not expose feature_importances_.")

    st.markdown("")
    st.subheader("Churn by Segment")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data for selected filters.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Churn Rate by Internet Service**")
            seg1 = (
                filtered_df.groupby("InternetService")["churn_label"]
                .mean().mul(100).reset_index()
                .rename(columns={"churn_label": "ChurnRate(%)"})
            )
            fig_int = px.bar(
                seg1,
                x="InternetService",
                y="ChurnRate(%)",
                template="glass_neon"
            )
            fig_int.update_layout(height=320)
            st.plotly_chart(fig_int, use_container_width=True)

        with c2:
            st.markdown("**Churn Rate by Payment Method**")
            seg2 = (
                filtered_df.groupby("PaymentMethod")["churn_label"]
                .mean().mul(100).reset_index()
                .rename(columns={"churn_label": "ChurnRate(%)"})
            )
            fig_pay = px.bar(
                seg2,
                x="PaymentMethod",
                y="ChurnRate(%)",
                template="glass_neon"
            )
            fig_pay.update_layout(height=320)
            st.plotly_chart(fig_pay, use_container_width=True)

    # ============================================================
    # 🔥 TOP AT-RISK CUSTOMERS (Filtered)
    # ============================================================

    st.markdown("### 🔥 Top At-Risk Customers (Filtered)")

    top_risk = filtered_df.sort_values("churn_prob", ascending=False).head(10)

    show_df = top_risk[[
        "customerID", "tenure", "MonthlyCharges", "Contract",
        "InternetService", "churn_prob", "risk_level"
    ]].copy()

    # Add retention recommendation
    show_df["Retention Recommendation"] = top_risk.apply(
        lambda r: retention_recommendation(r, short=True),
        axis=1
    )

    show_df["Risk Score (%)"] = (show_df["churn_prob"] * 100).round(2)

    # Badge renderer
    def badge(level):
        if level == "Low":
            return '<span class="badge-low">Low</span>'
        elif level == "Medium":
            return '<span class="badge-med">Medium</span>'
        else:
            return '<span class="badge-high">High</span>'

    show_df["Risk Level"] = show_df["risk_level"].apply(badge)

    show_df = show_df.drop(columns=["churn_prob", "risk_level"])

    # Highlight rows
    row_classes = []
    for i in range(len(show_df)):
        if i == 0:
            row_classes.append("row-top-1")
        elif i == 1:
            row_classes.append("row-top-2")
        elif i == 2:
            row_classes.append("row-top-3")
        else:
            row_classes.append("")
    show_df["row_class"] = row_classes

    # Build HTML table
    header_html = "<tr>" + "".join(
        f"<th>{col}</th>" for col in show_df.columns if col != "row_class"
    ) + "</tr>"

    body_html = ""
    for _, row in show_df.iterrows():
        cells = "".join(
            f"<td>{row[col]}</td>" for col in show_df.columns if col != "row_class"
        )
        body_html += f"<tr class='{row['row_class']}'>{cells}</tr>"

    html_table = f"""
    <table class="styled-table">
        <thead>{header_html}</thead>
        <tbody>{body_html}</tbody>
    </table>
    """

    st.markdown(
        f"<div class='glass-table'>{html_table}</div>",
        unsafe_allow_html=True
    )

    # ============================================================
    # 👤 CUSTOMER PROFILE VISUALIZATION
    # ============================================================

    st.markdown("### 👤 View Customer Details")

    selected = st.selectbox("🔎 Select a customer", top_risk["customerID"])

    if selected:
        cust = top_risk[top_risk["customerID"] == selected].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class='profile-card'>
                <h3>📌 Customer Overview</h3>
                <p><b>Customer ID:</b> {cust['customerID']}</p>
                <p><b>Contract:</b> {cust['Contract']}</p>
                <p><b>Internet:</b> {cust['InternetService']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='profile-card'>
                <h3>📊 Numerical Metrics</h3>
                <p><b>Tenure:</b> {cust['tenure']} months</p>
                <p><b>Monthly Charges:</b> ${cust['MonthlyCharges']}</p>
                <p><b>Risk Score:</b> {(cust['churn_prob']*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- RETENTION DETAILS ----------------
        full_recs = retention_recommendation(cust, short=False)

        st.markdown("""
         <div class='profile-card'>
             <h3>💡 Recommended Retention Actions</h3>
             <ul>
         """ + "".join(f"<li>{rec}</li>" for rec in full_recs) + """
             </ul>
        </div>
        """, unsafe_allow_html=True)


        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Tenure", "Monthly Charges", "Risk Score"],
            y=[cust["tenure"], cust["MonthlyCharges"], cust["churn_prob"]*100],
            marker=dict(color=['#22c55e','#3b82f6','#ef4444'])
        ))
        fig.update_layout(
            template="glass_neon",
            height=300,
            title="Customer Metric Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 – PREDICT CHURN (single customer)
# ============================================================
with tab_predict:
    st.subheader("Predict — Single Customer")
    if model is None:
        st.error("Model not available. Run training pipeline.")
    else:
        left, right = st.columns([1,1])
        
        # ---------------- LEFT COLUMN ----------------
        with left:
            gender = st.selectbox("Gender", ["Male","Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0,1])
            partner = st.selectbox("Partner", ["Yes","No"])
            dependents = st.selectbox("Dependents", ["Yes","No"])
            phone_service = st.selectbox("Phone Service", ["Yes","No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service","No","Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

        # ---------------- RIGHT COLUMN ----------------
        with right:
            online_security = st.selectbox("Online Security", ["No internet service","No","Yes"])
            online_backup = st.selectbox("Online Backup", ["No internet service","No","Yes"])
            device_protection = st.selectbox("Device Protection", ["No internet service","No","Yes"])
            tech_support = st.selectbox("Tech Support", ["No internet service","No","Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No internet service","No","Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No internet service","No","Yes"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes","No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
            )

        # ---------------- NUMERIC INPUTS ----------------
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 50000.0, 1000.0)

        # ---------------- PREDICTION ----------------
        if st.button("Predict Churn Risk"):

            # Build input row
            input_df = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }])

            processed = preprocess_input(input_df)

            # # Probability
            # prob = float(model.predict_proba(processed)[0][1])
            # pred = 1 if prob > 0.5 else 0
            # risk_pct = prob * 100
            api_result = predict_via_backend(processed)
            if api_result:
                prob = api_result["churn_probability"]
                risk_pct = prob * 100
                pred = 1 if api_result["risk_level"] == "High" else 0
            else:
                prob = float(model.predict_proba(processed)[0][1])
                pred = 1 if prob > 0.5 else 0
                risk_pct = prob * 100



            # Risk styling
            if risk_pct < 33:
                pill = "pill-low"; lbl = "Low Risk"
            elif risk_pct < 66:
                pill = "pill-med"; lbl = "Medium Risk"
            else:
                pill = "pill-high"; lbl = "High Risk"

            indicator_left = min(max(risk_pct, 0), 100)

            # ---------------- UI BLOCK ----------------
            html = f"""
<div class="glass risk-bar-container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div style="font-weight:800;color:#e7fafd">Predicted churn probability</div>
    <div style="font-size:0.95rem;color:#9fd7dc">Model output</div>
  </div>

  <div class="risk-bar" role="progressbar" aria-valuenow="{risk_pct:.1f}">
    <div class="risk-indicator" style="left:{indicator_left}%;"></div>
  </div>

  <div style="margin-top:12px;display:flex;align-items:center;gap:18px;">
    <div style="font-size:1.6rem;color:#e6eef6;font-weight:800;">{risk_pct:.1f}%</div>
    <div class="risk-pill {pill}">{lbl}</div>
    <div style="color:#93bfc6;margin-left:12px;">
        Model Prediction: <b style="color:#fff">{'Will Churn' if pred==1 else 'Will Not Churn'}</b>
    </div>
  </div>
</div>
"""
            st.markdown(html, unsafe_allow_html=True)

            # Optional Feature Importance
            if hasattr(model, "feature_importances_") and columns is not None:
                fi = pd.DataFrame({
                    "feature": columns,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False).head(8)

                fig = px.bar(
                    fi,
                    x="importance", y="feature",
                    orientation="h",
                    template="glass_neon",
                    height=320
                )
                st.plotly_chart(fig, use_container_width=True)
# ============================================================
# TAB 4 – MODEL PERFORMANCE
# ============================================================
with tab_performance:

    st.header("📊 Model Performance Overview")

    # ----------------------------
    # Load metrics.json
    # ----------------------------
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = None

    # ----------------------------
    # Best Model Summary
    # ----------------------------
    st.subheader("🏆 Best Model Summary")

    if metrics_data:
        best_model = metrics_data.get("best_model", "N/A")
        best_metric_name = metrics_data.get("best_metric_name", "N/A")
        best_metric_value = metrics_data.get("best_metric_value", 0)

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Best Model", best_model)

        with c2:
            st.metric(
                f"Best {best_metric_name.upper()}",
                f"{best_metric_value:.4f}"
            )
    else:
        st.warning("⚠ No metrics found. Please run training pipeline.")

    st.markdown("---")

    # ----------------------------
    # Leaderboard
    # ----------------------------
    st.subheader("📈 Model Leaderboard")

    if os.path.exists(LEADERBOARD_PATH):
        leaderboard_df = pd.read_csv(LEADERBOARD_PATH)
        st.dataframe(leaderboard_df, use_container_width=True)
    else:
        st.info("Leaderboard not found. Train models to generate results.")

    st.markdown("---")

    # ----------------------------
    # Confusion Matrices
    # ----------------------------
    st.subheader("📉 Confusion Matrices")

    if os.path.exists(REPORTS_DIR):
        cm_files = sorted([f for f in os.listdir(REPORTS_DIR) if f.startswith("cm_")])

        if cm_files:
            cols = st.columns(3)
            for i, file in enumerate(cm_files):
                with cols[i % 3]:
                    st.image(
                        os.path.join(REPORTS_DIR, file),
                        caption=file.replace("cm_", "").replace(".png", "")
                    )
        else:
            st.info("No confusion matrices available. Train models first.")
    else:
        st.info("Reports directory not found. Train models to generate reports.")

    st.markdown("---")

    # ----------------------------
    # ROC Curves
    # ----------------------------
    st.subheader("📈 ROC Curves")

    if os.path.exists(REPORTS_DIR):
        roc_files = sorted([f for f in os.listdir(REPORTS_DIR) if f.startswith("roc_")])

        if roc_files:
            cols = st.columns(3)
            for i, file in enumerate(roc_files):
                with cols[i % 3]:
                    st.image(
                        os.path.join(REPORTS_DIR, file),
                        caption=file.replace("roc_", "").replace(".png", "")
                    )
        else:
            st.info("No ROC curves available. Train models first.")
    else:
        st.info("Reports directory not found. Train models to generate reports.")
