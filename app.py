import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL UI STYLE
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.15), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(139,92,246,0.14), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #0b1728 50%, #101827 100%);
}

/* Main content */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101d 0%, #0b1422 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5edf8;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1 !important;
}

/* Header */
.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(30,64,175,0.35),
        rgba(124,58,237,0.25)
    );
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #cbd5e1;
}

/* Section headers */
.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 20px;
    margin-bottom: 12px;
}

/* Inputs */
div[data-baseweb="input"],
div[data-baseweb="select"] {
    background: rgba(15,23,42,0.75);
    border-radius: 10px;
}

input, textarea {
    color: #f8fafc !important;
}

/* Select text */
div[data-baseweb="select"] * {
    color: #f8fafc !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 13px;
    padding: 14px 20px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: 0 10px 25px rgba(37,99,235,0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(124,58,237,0.35);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 18px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 15px;
}

/* Horizontal line */
hr {
    border-color: rgba(255,255,255,0.10);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("churn_model.json")
    return model


model = load_model()

# Get feature names directly from the trained XGBoost model
model_features = model.get_booster().feature_names

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("## 📊 ChurnAI")

    st.caption("AI-Powered Customer Churn Prediction")

    st.divider()

    st.markdown("### 🧠 Model")

    st.info(
        "XGBoost Classification Model\n\n"
        "Designed to identify customers "
        "who may discontinue the service."
    )

    st.markdown("### ⚙️ Model Details")

    st.metric("Algorithm", "XGBoost")
    st.metric("Prediction Type", "Binary Classification")

    st.divider()

    st.markdown("### 📌 Input Categories")

    st.write("👤 Customer Profile")
    st.write("🌐 Services")
    st.write("💳 Billing")
    st.write("📅 Contract")

    st.divider()

    st.caption("Customer Churn Analysis Project")
    st.caption("AI/ML • XGBoost")

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">📊 Customer Churn Prediction</div>
    <div class="hero-subtitle">
        Intelligent customer retention analysis powered by an XGBoost machine learning model.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CUSTOMER PROFILE
# --------------------------------------------------
st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col4:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

# --------------------------------------------------
# CUSTOMER TENURE
# --------------------------------------------------
st.markdown('<div class="section-title">📅 Customer Relationship</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )

# --------------------------------------------------
# SERVICES
# --------------------------------------------------
st.markdown('<div class="section-title">🌐 Service Information</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col4:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col4:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

col1, col2 = st.columns(2)

with col1:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

# --------------------------------------------------
# BILLING
# --------------------------------------------------
st.markdown('<div class="section-title">💳 Billing Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------
st.divider()

_, button_col, _ = st.columns([1, 2, 1])

with button_col:

    analyze = st.button(
        "🔍  ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if analyze:

    # --------------------------------------------------
    # CREATE CUSTOMER DATA
    # --------------------------------------------------
    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # --------------------------------------------------
    # SAME ENCODING LOGIC
    # --------------------------------------------------
    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)

    # --------------------------------------------------
    # SAME MODEL PREDICTION
    # --------------------------------------------------
    prediction = model.predict(customer_encoded)[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------
    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ HIGH CHURN RISK")
        else:
            st.success("✅ CUSTOMER LIKELY TO STAY")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

    with result_col3:
        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.1f}%"
        )

    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------
    chart_col1, chart_col2 = st.columns([1.3, 1])

    with chart_col1:

        fig = go.Figure(
            go.Pie(
                labels=["Stay", "Churn"],
                values=[
                    stay_probability,
                    probability
                ],
                hole=0.65,
                textinfo="label+percent",
                marker=dict(
                    line=dict(
                        color="#0b1728",
                        width=3
                    )
                )
            )
        )

        fig.update_layout(
            title="Customer Churn Probability",
            template="plotly_dark",
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart_col2:

        st.markdown("### 📌 Prediction Summary")

        if prediction == 1:

            st.warning(
                "This customer has been classified as having "
                "a higher probability of churn by the XGBoost model."
            )

        else:

            st.success(
                "This customer has been classified as having "
                "a higher probability of staying with the service."
            )

        st.progress(
            float(probability)
        )

        st.caption(
            f"Model confidence toward churn: "
            f"{probability * 100:.1f}%"
        )

    # --------------------------------------------------
    # CUSTOMER SUMMARY
    # --------------------------------------------------
    with st.expander("🔎 View Customer Input Summary"):

        summary = pd.DataFrame({
            "Feature": [
                "Gender",
                "Senior Citizen",
                "Partner",
                "Dependents",
                "Tenure",
                "Phone Service",
                "Multiple Lines",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Contract",
                "Paperless Billing",
                "Payment Method",
                "Monthly Charges",
                "Total Charges"
            ],
            "Value": [
                gender,
                senior_citizen,
                partner,
                dependents,
                f"{tenure} months",
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                f"${monthly_charges:.2f}",
                f"${total_charges:.2f}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Customer Churn Analysis • XGBoost Machine Learning Model • AI/ML Project
</div>
""", unsafe_allow_html=True)
