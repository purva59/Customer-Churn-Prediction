import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ChurnAI - Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LIGHT PROFESSIONAL DESIGN
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        linear-gradient(135deg, #f7fbff 0%, #eef5ff 45%, #f8f5ff 100%);
}

/* Main area */
.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5eaf2;
}

section[data-testid="stSidebar"] * {
    color: #24324a;
}

/* Hero */
.hero-box {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #eef5ff 55%,
        #f5f0ff 100%
    );
    border: 1px solid #dce7f7;
    border-radius: 28px;
    padding: 34px 40px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(55, 84, 130, 0.10);
}

.hero-small {
    color: #4774a8;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.hero-title {
    color: #172b4d;
    font-size: 40px;
    font-weight: 800;
    margin-top: 8px;
    margin-bottom: 8px;
}

.hero-description {
    color: #63748b;
    font-size: 16px;
}

/* Section title */
.section-heading {
    color: #172b4d;
    font-size: 22px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 14px;
}

/* Cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e1e8f2;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 25px rgba(50, 75, 110, 0.07);
}

/* Input boxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #ffffff;
    border-color: #d8e1ed;
    border-radius: 10px;
}

/* Text inside inputs */
div[data-baseweb="select"] span {
    color: #26364f !important;
}

input {
    color: #26364f !important;
}

/* Button */
.stButton > button {
    border: none;
    border-radius: 14px;
    min-height: 54px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(
        90deg,
        #3978e8,
        #6954e8
    );
    box-shadow: 0 10px 24px rgba(67, 93, 220, 0.25);
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(67, 93, 220, 0.32);
}

/* Result containers */
.result-good {
    background: #ecfdf5;
    border: 1px solid #b7ebd2;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.result-risk {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
}

.result-subtitle {
    font-size: 14px;
    margin-top: 8px;
}

/* Information cards */
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(50, 75, 110, 0.06);
}

/* Expander */
div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e1e8f2;
    border-radius: 15px;
}

/* Divider */
hr {
    border-color: #e1e8f2;
}

/* Footer */
.footer {
    text-align: center;
    color: #8492a6;
    font-size: 13px;
    padding: 30px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD YOUR EXISTING XGBOOST MODEL
# =========================================================
@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

# Get feature names directly from your trained model
model_features = model.get_booster().feature_names


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🤖 ChurnAI")

    st.caption("Customer Churn Intelligence")

    st.divider()

    st.markdown("### 🧠 AI Model")

    st.write("**Algorithm**")
    st.write("XGBoost Classifier")

    st.write("**Task**")
    st.write("Binary Classification")

    st.write("**Output**")
    st.write("Churn Probability")

    st.divider()

    st.markdown("### 📋 Analysis Areas")

    st.write("👤 Customer Profile")
    st.write("🌐 Service Usage")
    st.write("💳 Billing Details")
    st.write("📅 Contract Information")

    st.divider()

    st.info(
        "This application uses a trained XGBoost "
        "model to estimate customer churn risk."
    )


# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero-box">

    <div class="hero-small">
        ✦ Artificial Intelligence • Customer Analytics
    </div>

    <div class="hero-title">
        🤖 Customer Churn Prediction
    </div>

    <div class="hero-description">
        Analyze customer information and estimate the probability
        of service churn using an XGBoost machine learning model.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# CUSTOMER PROFILE
# =========================================================
st.markdown(
    '<div class="section-heading">👤 Customer Profile</div>',
    unsafe_allow_html=True
)

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


# =========================================================
# CUSTOMER TENURE & CHARGES
# =========================================================
st.markdown(
    '<div class="section-heading">📈 Customer Value</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    tenure = st.number_input(
        "Tenure (Months)",
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


# =========================================================
# SERVICE INFORMATION
# =========================================================
st.markdown(
    '<div class="section-heading">🌐 Service Information</div>',
    unsafe_allow_html=True
)

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


# =========================================================
# BILLING
# =========================================================
st.markdown(
    '<div class="section-heading">💳 Billing & Payment</div>',
    unsafe_allow_html=True
)

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


# =========================================================
# ANALYZE BUTTON
# =========================================================
st.divider()

button_left, button_middle, button_right = st.columns(
    [1, 2, 1]
)

with button_middle:

    analyze = st.button(
        "🔮  PREDICT CUSTOMER CHURN",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if analyze:

    # -----------------------------------------------------
    # CREATE CUSTOMER DATA
    # -----------------------------------------------------

    customer = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

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


    # -----------------------------------------------------
    # SAME ENCODING USED FOR YOUR MODEL
    # -----------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)


    # -----------------------------------------------------
    # YOUR ORIGINAL XGBOOST PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">🎯 AI Prediction</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # MAIN RESULT
    # -----------------------------------------------------

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-risk">

                <div class="result-title">
                    ⚠️ Customer At Risk
                </div>

                <div class="result-subtitle">
                    The model predicts a higher probability
                    of customer churn.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-good">

                <div class="result-title">
                    ✅ Customer Likely to Stay
                </div>

                <div class="result-subtitle">
                    The model predicts a higher probability
                    of customer retention.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # =====================================================
    # PROBABILITY METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.1f}%"
        )

    with col3:

        if probability >= 0.5:

            risk_text = "Higher Risk"

        else:

            risk_text = "Lower Risk"

        st.metric(
            "Risk Status",
            risk_text
        )


    # =====================================================
    # CHART
    # =====================================================

    st.markdown(
        '<div class="section-heading">📊 Prediction Analysis</div>',
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns(
        [1.25, 1]
    )


    with chart_col1:

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Stay", "Churn"],
                y=[
                    stay_probability * 100,
                    probability * 100
                ],
                text=[
                    f"{stay_probability * 100:.1f}%",
                    f"{probability * 100:.1f}%"
                ],
                textposition="outside",
                marker=dict(
                    color=[
                        "#4f9cf9",
                        "#8b6cf6"
                    ],
                    line=dict(
                        width=0
                    )
                )
            )
        )

        fig.update_layout(

            title="Customer Probability",

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            xaxis=dict(
                title=""
            ),

            height=390,

            template="plotly_white",

            margin=dict(
                l=40,
                r=30,
                t=70,
                b=40
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with chart_col2:

        st.markdown(
            """
            <div class="info-card">

            <h3 style="color:#172b4d;">
            🔍 Prediction Insight
            </h3>

            <p style="color:#63748b;">
            The XGBoost model evaluates the customer's
            service, contract, tenure and billing
            information to generate a churn probability.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.progress(
            float(probability)
        )

        st.caption(
            f"Churn probability: "
            f"{probability * 100:.1f}%"
        )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    with st.expander("📋 View Customer Information"):

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


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🤖 ChurnAI &nbsp;•&nbsp;
        Customer Churn Analysis &nbsp;•&nbsp;
        XGBoost Machine Learning

    </div>
    """,
    unsafe_allow_html=True
)
