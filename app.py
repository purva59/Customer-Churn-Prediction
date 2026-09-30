import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnAI | Customer Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL LIGHT UI
# =========================================================

st.markdown("""
<style>

/* ---------- PAGE ---------- */

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(75, 137, 220, 0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 100% 0%,
            rgba(124, 92, 220, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 48%,
            #f8f6ff 100%
        );
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e4eaf3;
}

section[data-testid="stSidebar"] h1 {
    color: #183b67 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #64748b !important;
}


/* ---------- HEADINGS ---------- */

h1 {
    color: #16395f !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    color: #214a76 !important;
    font-weight: 750 !important;
}

h3 {
    color: #2d5278 !important;
    font-weight: 700 !important;
}


/* ---------- NORMAL TEXT ---------- */

p {
    color: #61738b;
}


/* ---------- INPUTS ---------- */

div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border: 1px solid #d8e2ef;
    border-radius: 11px;
    min-height: 42px;
}

div[data-baseweb="input"] > div {
    background-color: #ffffff;
    border: 1px solid #d8e2ef;
    border-radius: 11px;
    min-height: 42px;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    width: 100%;
    min-height: 58px;
    border: none;
    border-radius: 15px;

    background: linear-gradient(
        100deg,
        #2878df,
        #6654dc
    );

    color: white;
    font-size: 17px;
    font-weight: 800;

    box-shadow:
        0 12px 28px rgba(60, 91, 210, 0.24);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 16px 34px rgba(60, 91, 210, 0.32);
}


/* ---------- METRICS ---------- */

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid #e1e8f1;
    border-radius: 18px;

    padding: 20px;

    box-shadow:
        0 8px 25px rgba(44, 74, 112, 0.08);
}


/* ---------- EXPANDER ---------- */

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e0e7f0;
    border-radius: 15px;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: #dfe7f1;
}


/* ---------- PROGRESS ---------- */

div[data-testid="stProgress"] {
    border-radius: 10px;
}


/* ---------- DATAFRAME ---------- */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}


/* ---------- ALERTS ---------- */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ---------- MOBILE ---------- */

@media (max-width: 800px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 30px !important;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD EXISTING XGBOOST MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

# Exact features from trained model
model_features = model.get_booster().feature_names


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 ChurnAI")

    st.caption("Customer Intelligence Platform")

    st.divider()

    st.subheader("🧠 AI Model")

    st.write("**Algorithm**")
    st.write("XGBoost Classifier")

    st.write("**Model Type**")
    st.write("Binary Classification")

    st.write("**Prediction**")
    st.write("Customer Churn")

    st.divider()

    st.subheader("📌 Dashboard")

    st.write("👤 Customer Profile")
    st.write("📈 Customer Value")
    st.write("🌐 Service Details")
    st.write("💳 Billing Details")
    st.write("🎯 AI Prediction")

    st.divider()

    st.success(
        "Model loaded successfully"
    )

    st.caption(
        "Customer Churn Analysis • AI/ML Project"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 Customer Churn Prediction")

st.write(
    "An intelligent customer analytics application powered by "
    "an XGBoost machine learning model."
)

st.info(
    "💡 Enter the customer's details below. "
    "The trained model will analyze the information and "
    "generate a churn probability."
)


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.header("👤 Customer Profile")

st.caption(
    "Basic information about the customer"
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
# CUSTOMER VALUE
# =========================================================

st.header("📈 Customer Value")

st.caption(
    "Customer relationship duration and financial information"
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

st.header("🌐 Service Information")

st.caption(
    "Services currently used by the customer"
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
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


# =========================================================
# BILLING
# =========================================================

st.header("💳 Billing & Payment")

st.caption(
    "Payment and billing preferences"
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
# PREDICTION BUTTON
# =========================================================

st.divider()

st.subheader("🚀 Generate AI Prediction")

st.caption(
    "Click the button below to analyze this customer."
)

left, center, right = st.columns([1, 2, 1])

with center:

    predict = st.button(
        "🔮  ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # CUSTOMER DATA
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
    # ENCODING
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
    # EXISTING MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.header("🎯 AI Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ HIGHER CHURN RISK\n\n"
            "The trained XGBoost model predicts a higher "
            "probability that this customer may churn."
        )

    else:

        st.success(
            "✅ LOWER CHURN RISK\n\n"
            "The trained XGBoost model predicts a higher "
            "probability that this customer will stay."
        )


    # =====================================================
    # KEY RESULTS
    # =====================================================

    st.subheader("📊 Key Prediction Metrics")

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

        if prediction == 1:

            risk = "Higher Risk"

        else:

            risk = "Lower Risk"

        st.metric(
            "Risk Status",
            risk
        )


    # =====================================================
    # ANALYSIS
    # =====================================================

    st.subheader("📈 Prediction Analysis")

    chart_col, insight_col = st.columns(
        [1.4, 1]
    )


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    with chart_col:

        fig = go.Figure()

        fig.add_trace(
            go.Bar(

                x=[
                    "Stay",
                    "Churn"
                ],

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
                        "#4F8FEF",
                        "#8B6FE8"
                    ]
                )
            )
        )

        fig.update_layout(

            title="Customer Churn Probability",

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            xaxis=dict(
                title=""
            ),

            height=400,

            template="plotly_white",

            margin=dict(
                l=40,
                r=30,
                t=65,
                b=40
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # INSIGHT
    # -----------------------------------------------------

    with insight_col:

        st.subheader("🔍 Model Insight")

        st.write(
            "The XGBoost model analyzes the customer's "
            "profile, services, contract and billing "
            "information to produce the prediction."
        )

        st.write("")

        st.write("**Churn Probability**")

        st.progress(
            float(probability)
        )

        st.write(
            f"{probability * 100:.1f}%"
        )

        st.write("**Stay Probability**")

        st.progress(
            float(stay_probability)
        )

        st.write(
            f"{stay_probability * 100:.1f}%"
        )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.subheader("📋 Customer Information")

    with st.expander(
        "View Customer Details"
    ):

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

st.divider()

st.caption(
    "🤖 ChurnAI  |  Customer Churn Intelligence  |  "
    "Powered by XGBoost"
)
