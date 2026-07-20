import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Insurance Claim Fraud Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS  (visual layer only — no logic here)
# =========================================================
st.markdown("""
<style>

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* ---------- Hero Section ---------- */
    .hero-container {
        background: linear-gradient(135deg, #1a1f2b 0%, #131722 100%);
        border: 1px solid #232838;
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #b8bfcc;
        max-width: 850px;
        line-height: 1.5;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border: 1px solid transparent;
    }

    .badge-blue   { background: rgba(56,132,255,0.12); color: #6fa8ff; border-color: rgba(56,132,255,0.35); }
    .badge-green  { background: rgba(43,196,127,0.12); color: #4ade80; border-color: rgba(43,196,127,0.35); }
    .badge-purple { background: rgba(168,85,247,0.12); color: #c084fc; border-color: rgba(168,85,247,0.35); }
    .badge-orange { background: rgba(255,159,67,0.12); color: #ffb454; border-color: rgba(255,159,67,0.35); }

    /* ---------- Section Cards ---------- */
    .section-card {
        background: #151a24;
        border: 1px solid #232838;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        border-radius: 16px;
        padding: 1.3rem 1.2rem;
        text-align: center;
        border: 1px solid #232838;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        transition: transform 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9aa3b5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
    }

    .metric-fraud   { background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.03)); }
    .metric-genuine { background: linear-gradient(135deg, rgba(43,196,127,0.15), rgba(43,196,127,0.03)); }
    .metric-confidence { background: linear-gradient(135deg, rgba(56,132,255,0.15), rgba(56,132,255,0.03)); }

    /* ---------- Result Banner ---------- */
    .result-banner {
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        text-align: center;
        border: 1px solid;
    }

    .result-fraud {
        background: rgba(239,68,68,0.10);
        border-color: rgba(239,68,68,0.4);
        color: #fca5a5;
    }

    .result-genuine {
        background: rgba(43,196,127,0.10);
        border-color: rgba(43,196,127,0.4);
        color: #86efac;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35);
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(37,99,235,0.5);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #10141c;
        border-right: 1px solid #232838;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.1rem;
    }

    .sidebar-caption {
        font-size: 0.82rem;
        color: #8b93a5;
        margin-bottom: 1rem;
    }

    /* ---------- Explanation Card ---------- */
    .explain-card {
        background: #151a24;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.6rem;
    }

    /* ---------- Divider spacing ---------- */
    hr {
        margin: 1.4rem 0;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA / MODEL LOADING  (unchanged logic — cached only for UX)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model_assets():
    model = joblib.load("best_xgb.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    explainer = shap.TreeExplainer(model)
    return model, label_encoders, explainer


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv("carclaims.csv")


with st.spinner("Loading model and dataset..."):
    model, label_encoders, explainer = load_model_assets()
    df = load_dataset()

# -----------------------------
# Load Dataset Information (unchanged)
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

# Remove target variable
if "FraudFound" in categorical_cols:
    categorical_cols.remove("FraudFound")

if "FraudFound" in numerical_cols:
    numerical_cols.remove("FraudFound")

# Store possible values for each categorical feature
category_options = {}

for col in categorical_cols:
    category_options[col] = sorted(df[col].dropna().unique().tolist())

# Feature order used during training (moved to module scope — same values,
# just fixed so it is always defined; no logic change)
feature_order = [
    'Month',
    'WeekOfMonth',
    'DayOfWeek',
    'Make',
    'AccidentArea',
    'DayOfWeekClaimed',
    'MonthClaimed',
    'WeekOfMonthClaimed',
    'Sex',
    'MaritalStatus',
    'Age',
    'Fault',
    'PolicyType',
    'VehicleCategory',
    'VehiclePrice',
    'PolicyNumber',
    'RepNumber',
    'Deductible',
    'DriverRating',
    'Days:Policy-Accident',
    'Days:Policy-Claim',
    'PastNumberOfClaims',
    'AgeOfVehicle',
    'AgeOfPolicyHolder',
    'PoliceReportFiled',
    'WitnessPresent',
    'AgentType',
    'NumberOfSuppliments',
    'AddressChange-Claim',
    'NumberOfCars',
    'Year',
    'BasePolicy'
]

# =========================================================
# SESSION STATE
# =========================================================
if "result" not in st.session_state:
    st.session_state.result = None

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🚗 FraudGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Insurance Claim Fraud Detection</div>', unsafe_allow_html=True)

    st.divider()

    with st.expander("ℹ️ About this Project", expanded=True):
        st.write(
            "FraudGuard AI analyzes insurance claim details and predicts "
            "whether a claim is likely to be **fraudulent** or **genuine**, "
            "using a trained machine learning model with explainable AI "
            "insights for every prediction."
        )

    with st.expander("🧠 Model Information"):
        st.markdown("""
        - **Algorithm:** XGBoost Classifier
        - **Accuracy:** 95.07%
        - **ROC-AUC:** 0.936
        - **Explainability:** SHAP
        """)

    with st.expander("📁 Dataset Information"):
        st.markdown(f"""
        - **Rows:** {df.shape[0]:,}
        - **Columns:** {df.shape[1]}
        - **Categorical Features:** {len(categorical_cols)}
        - **Numerical Features:** {len(numerical_cols)}
        """)

    with st.expander("🛠️ Technology Stack"):
        st.markdown("""
        - Streamlit — UI framework
        - XGBoost — ML model
        - SHAP — Explainable AI
        - Pandas — Data processing
        - Scikit-learn — Preprocessing
        """)

    st.divider()
    st.caption("Built for academic & portfolio demonstration purposes.")

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🚗 Insurance Claim Fraud Detection System</div>
    <div class="hero-subtitle">
        An AI-powered dashboard that uses a trained <b>XGBoost</b> model to assess insurance claims
        and flag potential fraud in real time, backed by <b>SHAP explainable AI</b> so every decision
        can be understood, not just trusted.
    </div>
    <div style="margin-top: 1rem;">
        <span class="badge badge-blue">🎯 Accuracy: 95.07%</span>
        <span class="badge badge-purple">📈 ROC-AUC: 0.936</span>
        <span class="badge badge-green">🔍 Explainable AI</span>
        <span class="badge badge-orange">🚨 Fraud Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN TABS
# =========================================================
tab_input, tab_results = st.tabs(["📝 Claim Details", "📊 Prediction Results"])

# ---------------------------------------------------------
# TAB 1 — CLAIM INPUT FORM
# ---------------------------------------------------------
with tab_input:

    user_input = {}

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📋 Policy Information")
    st.caption("Basic details about the insurance policy tied to this claim.")

    col1, col2 = st.columns(2)

    policy_features = [
        "Month",
        "WeekOfMonth",
        "PolicyType",
        "BasePolicy",
        "PolicyNumber",
        "Deductible",
        "Year"
    ]

    for i, feature in enumerate(policy_features):
        container = col1 if i % 2 == 0 else col2
        with container:
            if feature in categorical_cols:
                user_input[feature] = st.selectbox(feature, category_options[feature], key=f"in_{feature}")
            else:
                user_input[feature] = st.number_input(
                    feature, value=float(df[feature].median()), key=f"in_{feature}"
                )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🚗 Vehicle Information")
    st.caption("Details describing the insured vehicle.")

    col1, col2 = st.columns(2)

    vehicle_features = [
        "Make",
        "VehicleCategory",
        "VehiclePrice",
        "AgeOfVehicle",
        "NumberOfCars"
    ]

    for i, feature in enumerate(vehicle_features):
        container = col1 if i % 2 == 0 else col2
        with container:
            if feature in categorical_cols:
                user_input[feature] = st.selectbox(feature, category_options[feature], key=f"in_{feature}")
            else:
                user_input[feature] = st.number_input(
                    feature, value=float(df[feature].median()), key=f"in_{feature}"
                )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👤 Customer Information")
    st.caption("Details about the policyholder.")

    col1, col2 = st.columns(2)

    customer_features = [
        "Age",
        "Sex",
        "MaritalStatus",
        "AgeOfPolicyHolder",
        "DriverRating"
    ]

    for i, feature in enumerate(customer_features):
        container = col1 if i % 2 == 0 else col2
        with container:
            if feature in categorical_cols:
                user_input[feature] = st.selectbox(feature, category_options[feature], key=f"in_{feature}")
            else:
                user_input[feature] = st.number_input(
                    feature, value=float(df[feature].median()), key=f"in_{feature}"
                )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🚨 Claim Information")
    st.caption("Circumstances and history surrounding the claim itself.")

    col1, col2 = st.columns(2)

    claim_features = [
        "DayOfWeek",
        "AccidentArea",
        "DayOfWeekClaimed",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "Fault",
        "RepNumber",
        "Days:Policy-Accident",
        "Days:Policy-Claim",
        "PastNumberOfClaims",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "NumberOfSuppliments",
        "AddressChange-Claim"
    ]

    for i, feature in enumerate(claim_features):
        container = col1 if i % 2 == 0 else col2
        with container:
            if feature in categorical_cols:
                user_input[feature] = st.selectbox(feature, category_options[feature], key=f"in_{feature}")
            else:
                user_input[feature] = st.number_input(
                    feature, value=float(df[feature].median()), key=f"in_{feature}"
                )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    predict = st.button("🔍 Predict Fraud", use_container_width=True)

    if predict:
        with st.status("Running fraud analysis...", expanded=True) as status:

            st.write("Encoding claim details...")
            # Convert user input to DataFrame
            input_df = pd.DataFrame([user_input])

            # Encode categorical columns
            for col in categorical_cols:
                input_df[col] = label_encoders[col].transform(input_df[col])

            # Arrange columns in the same order used during training
            input_df = input_df[feature_order]

            st.write("Running XGBoost prediction...")
            # Prediction
            prediction = model.predict(input_df)[0]

            # Prediction probability
            probability = model.predict_proba(input_df)[0]

            fraud_probability = probability[1] * 100
            genuine_probability = probability[0] * 100
            confidence = max(probability) * 100

            st.write("Computing SHAP explanations...")
            shap_values = explainer(input_df)

            # -----------------------------
            # Decode categorical values
            # -----------------------------
            display_values = []

            for col in input_df.columns:
                value = input_df[col].iloc[0]
                if col in label_encoders:
                    value = label_encoders[col].inverse_transform([int(value)])[0]
                display_values.append(value)

            # Create SHAP importance dataframe
            importance_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Value": display_values,
                "SHAP Impact": shap_values.values[0]
            })

            importance_df["Absolute Impact"] = importance_df["SHAP Impact"].abs()
            importance_df = importance_df.sort_values(by="Absolute Impact", ascending=False)
            top5 = importance_df.head(5)

            status.update(label="Analysis complete", state="complete", expanded=False)

        # Persist everything needed to render the Results tab
        st.session_state.result = {
            "prediction": prediction,
            "fraud_probability": fraud_probability,
            "genuine_probability": genuine_probability,
            "confidence": confidence,
            "shap_values": shap_values,
            "top5": top5,
        }

        st.success("✅ Prediction complete — see the **Prediction Results** tab.")

# ---------------------------------------------------------
# TAB 2 — RESULTS
# ---------------------------------------------------------
with tab_results:

    if st.session_state.result is None:
        st.info("👈 Fill in the claim details and click **Predict Fraud** to see results here.")
    else:
        r = st.session_state.result
        prediction = r["prediction"]
        fraud_probability = r["fraud_probability"]
        genuine_probability = r["genuine_probability"]
        confidence = r["confidence"]
        shap_values = r["shap_values"]
        top5 = r["top5"]

        # -----------------------------
        # Result Banner
        # -----------------------------
        if prediction == 1:
            st.markdown(
                '<div class="result-banner result-fraud">🚨 Fraudulent Claim Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-banner result-genuine">✅ Genuine Claim</div>',
                unsafe_allow_html=True
            )

        # -----------------------------
        # Metric Cards
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card metric-fraud">
                <div class="metric-label">Fraud Probability</div>
                <div class="metric-value">{fraud_probability:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card metric-genuine">
                <div class="metric-label">Genuine Probability</div>
                <div class="metric-value">{genuine_probability:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card metric-confidence">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # -----------------------------
        # SHAP Explanation
        # -----------------------------
        st.subheader("🔍 Explainable AI (SHAP)")
        st.markdown("""
        <div class="explain-card">
        <b>What is SHAP?</b><br>
        SHAP (SHapley Additive exPlanations) shows how much each claim detail pushed the
        prediction toward <b>Fraudulent</b> or <b>Genuine</b>. Bars pointing one way increase
        the fraud likelihood; bars pointing the other way reduce it.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # -----------------------------
        # Top 5 Feature Contributions
        # -----------------------------
        st.subheader("📋 Top Factors Influencing the Prediction")

        display_table = top5[["Feature", "Value", "SHAP Impact"]].rename(columns={
            "Feature": "Claim Attribute",
            "Value": "Submitted Value",
            "SHAP Impact": "Impact on Fraud Score"
        }).reset_index(drop=True)

        def _color_impact(val):
            color = "#4ade80" if val < 0 else "#f87171"
            return f"color: {color}; font-weight: 700;"

        styled_table = display_table.style \
            .format({"Impact on Fraud Score": "{:+.4f}"}) \
            .applymap(_color_impact, subset=["Impact on Fraud Score"])

        st.dataframe(styled_table, use_container_width=True)

        st.divider()

        # -----------------------------
        # Model Explanation
        # -----------------------------
        st.subheader("📝 Why did the model make this prediction?")

        prediction_text = "Fraudulent" if prediction == 1 else "Genuine"

        st.markdown(f"""
        <div class="explain-card">
        The model classified this claim as <b>{prediction_text}</b> because the following
        details had the strongest influence on the outcome:
        </div>
        """, unsafe_allow_html=True)

        for _, row in top5.iterrows():
            increasing = row["SHAP Impact"] > 0
            direction = "⬆ Increased fraud likelihood" if increasing else "⬇ Reduced fraud likelihood"
            color = "#f87171" if increasing else "#4ade80"

            st.markdown(f"""
            <div class="explain-card">
            <b>{row['Feature']}</b> = {row['Value']} &nbsp;→&nbsp;
            <span style="color:{color}; font-weight:700;">{direction}</span>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "Overall, the model combines the influence of all features before making its "
            "final prediction. Features with larger SHAP values have a stronger impact on "
            "the decision."
        )
