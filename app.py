import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LIGHT PROFESSIONAL THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #f7f3ff 100%
        );
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5eaf2;
    }

    section[data-testid="stSidebar"] * {
        color: #26364f;
    }

    h1 {
        color: #172b4d !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #203653 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #29415f !important;
        font-weight: 700 !important;
    }

    p {
        color: #64748b;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(50, 75, 110, 0.08);
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border-color: #d9e2ef;
    }

    div[data-baseweb="input"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border-color: #d9e2ef;
    }

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 13px;
        border: none;
        background: linear-gradient(
            90deg,
            #3978e8,
            #7055e8
        );
        color: white;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 8px 20px rgba(70, 90, 220, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #2869da,
            #6045d8
        );
        color: white;
    }

    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e1e8f2;
        border-radius: 14px;
    }

    hr {
        border-color: #dfe7f1;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

# Get feature names directly from trained model
model_features = model.get_booster().feature_names


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 ChurnAI")

    st.caption("Customer Churn Intelligence")

    st.divider()

    st.subheader("🧠 Model")

    st.write("**Algorithm:** XGBoost")

    st.write("**Task:** Binary Classification")

    st.write("**Output:** Churn Probability")

    st.divider()

    st.subheader("📋 Analysis Areas")

    st.write("👤 Customer Profile")

    st.write("📈 Customer Value")

    st.write("🌐 Service Information")

    st.write("💳 Billing & Payment")

    st.divider()

    st.info(
        "This application uses a trained XGBoost "
        "model to estimate customer churn risk."
    )

    st.caption("AI / ML Project")


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 Customer Churn Prediction")

st.caption(
    "AI-powered customer churn analysis using an XGBoost "
    "machine learning model."
)

st.info(
    "Enter the customer's information below and click "
    "**Predict Customer Churn** to generate the prediction."
)


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.header("👤 Customer Profile")

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
# PREDICT BUTTON
# =========================================================

st.divider()

left, center, right = st.columns([1, 2, 1])

with center:

    analyze = st.button(
        "🔮 Predict Customer Churn",
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
    # ENCODE CUSTOMER DATA
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
    # ORIGINAL XGBOOST PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability


    # =====================================================
    # AI PREDICTION
    # =====================================================

    st.divider()

    st.header("🎯 AI Prediction")

    if prediction == 1:

        st.error(
            "⚠️ Customer At Risk\n\n"
            "The model predicts a higher probability "
            "of customer churn."
        )

    else:

        st.success(
            "✅ Customer Likely to Stay\n\n"
            "The model predicts a higher probability "
            "of customer retention."
        )


    # =====================================================
    # PROBABILITY METRICS
    # =====================================================

    st.subheader("📊 Prediction Summary")

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

            risk_status = "Higher Risk"

        else:

            risk_status = "Lower Risk"

        st.metric(
            "Risk Status",
            risk_status
        )


    # =====================================================
    # CHURN PROBABILITY CHART
    # =====================================================

    st.subheader("📈 Prediction Analysis")

    chart_col1, chart_col2 = st.columns(
        [1.35, 1]
    )


    with chart_col1:

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

            title="Customer Probability",

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            xaxis=dict(
                title=""
            ),

            height=380,

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


    # =====================================================
    # PREDICTION INSIGHT
    # =====================================================

    with chart_col2:

        st.subheader("🔍 Prediction Insight")

        st.write(
            "The XGBoost model evaluates the customer's "
            "service, contract, tenure and billing "
            "information to generate a churn probability."
        )

        st.write("")

        st.progress(
            float(probability)
        )

        st.caption(
            f"Churn probability: "
            f"{probability * 100:.1f}%"
        )

        st.write("")

        if probability >= 0.5:

            st.warning(
                "The predicted churn probability is "
                "above 50%."
            )

        else:

            st.info(
                "The predicted churn probability is "
                "below 50%."
            )


    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.subheader("📋 Customer Information")

    with st.expander(
        "View Customer Input Details"
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
    "🤖 ChurnAI • Customer Churn Analysis • "
    "XGBoost Machine Learning"
)
