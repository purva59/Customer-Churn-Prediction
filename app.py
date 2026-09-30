import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL LIGHT UI
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f6f8fc;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- REMOVE DEFAULT TOP SPACE ---------- */

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- TITLE ---------- */

    h1 {
        color: #172554 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        color: #172554 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }

    /* ---------- TEXT ---------- */

    p {
        color: #475569;
    }

    /* ---------- INPUT LABELS ---------- */

    label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* ---------- SELECT BOX ---------- */

    div[data-baseweb="select"] > div {
        background: white !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 10px !important;
        min-height: 42px;
    }

    /* ---------- NUMBER INPUT ---------- */

    div[data-testid="stNumberInput"] input {
        background: white !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 10px !important;
        color: #1e293b !important;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            90deg,
            #2563eb,
            #4f46e5
        );
        color: white;
        font-size: 17px;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.20);
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            90deg,
            #1d4ed8,
            #4338ca
        );
        color: white;
        transform: translateY(-1px);
    }

    /* ---------- CARDS ---------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
    }

    /* ---------- METRIC ---------- */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 800 !important;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 25px 0;
    }

    /* ---------- PROGRESS ---------- */

    div[data-testid="stProgress"] > div > div {
        border-radius: 20px;
    }

    /* ---------- INFO BOXES ---------- */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


try:

    model = load_model()

except Exception as e:

    st.error("Model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# GET EXACT TRAINING FEATURES
# ============================================================

model_features = model.get_booster().feature_names

if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 Customer Churn AI")

st.markdown(
    "### Intelligent customer retention analysis powered by XGBoost"
)

st.write(
    "Enter customer information below to estimate the probability "
    "that the customer may discontinue the service."
)

st.divider()


# ============================================================
# TOP INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Machine Learning Model",
        "XGBoost"
    )

with info2:

    st.metric(
        "Prediction Type",
        "Binary Classification"
    )

with info3:

    st.metric(
        "Analysis",
        "Customer Churn"
    )


st.write("")


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Profile")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col3:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )


# ============================================================
# INTERNET & SERVICES
# ============================================================

st.write("")

st.subheader("🌐 Internet & Additional Services")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.write("")

st.subheader("💳 Contract & Billing")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col2:

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=10.0
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")
st.write("")

predict_button = st.button(
    "🔮  ANALYZE CUSTOMER CHURN",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE CUSTOMER DATA
    # --------------------------------------------------------

    customer = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet],

        "OnlineSecurity": [security],

        "OnlineBackup": [backup],

        "DeviceProtection": [protection],

        "TechSupport": [support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless],

        "PaymentMethod": [payment],

        "MonthlyCharges": [monthly],

        "TotalCharges": [total]
    })


    # --------------------------------------------------------
    # EXACT SAME ENCODING LOGIC
    # --------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        probability_percent = probability * 100


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.write("")
        st.divider()

        st.subheader("🎯 Prediction Result")


        # ----------------------------------------------------
        # RESULT STATUS
        # ----------------------------------------------------

        if prediction == 1:

            status_text = "HIGH CHURN RISK"

            status_description = (
                "The model predicts that this customer "
                "is likely to discontinue the service."
            )

            st.error(
                f"⚠️ {status_text}"
            )

        else:

            status_text = "LOW CHURN RISK"

            status_description = (
                "The model predicts that this customer "
                "is likely to continue the service."
            )

            st.success(
                f"✓ {status_text}"
            )


        st.write(status_description)


        # ====================================================
        # RESULT METRICS
        # ====================================================

        st.write("")

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Prediction",
                "CHURN" if prediction == 1 else "STAY"
            )

        with r2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

        with r3:

            st.metric(
                "Model",
                "XGBoost"
            )


        # ====================================================
        # PROBABILITY VISUALIZATION
        # ====================================================

        st.write("")
        st.subheader("📈 Churn Probability Analysis")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 42,
                        "color": "#172554"
                    }
                },
                title={
                    "text": "Estimated Churn Probability",
                    "font": {
                        "size": 18,
                        "color": "#475569"
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#94a3b8"
                    },
                    "bar": {
                        "color": "#2563eb",
                        "thickness": 0.75
                    },
                    "bgcolor": "#eef2f7",
                    "borderwidth": 1,
                    "bordercolor": "#dbe3ef"
                }
            )
        )

        gauge.update_layout(
            height=300,
            margin=dict(
                l=40,
                r=40,
                t=60,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={
                "family": "Arial"
            }
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        # ====================================================
        # CUSTOMER SUMMARY
        # ====================================================

        st.write("")
        st.subheader("📋 Customer Summary")

        summary1, summary2, summary3, summary4 = st.columns(4)

        with summary1:

            st.metric(
                "Contract",
                contract
            )

        with summary2:

            st.metric(
                "Tenure",
                f"{tenure} months"
            )

        with summary3:

            st.metric(
                "Monthly Charges",
                f"${monthly:.2f}"
            )

        with summary4:

            st.metric(
                "Internet",
                internet
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.write("")

        if prediction == 1:

            st.warning(
                "Retention attention may be appropriate for this "
                "customer based on the model prediction."
            )

        else:

            st.info(
                "The customer is currently classified as lower "
                "churn risk by the model."
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander("ℹ️ About this prediction"):

            st.write(
                "This application uses the trained XGBoost classification "
                "model to estimate customer churn probability."
            )

            st.write(
                "The prediction is generated from the customer information "
                "entered above and the same feature structure used by the "
                "trained model."
            )

            st.write(
                "The probability represents the model's estimated "
                "probability for the churn class."
            )


    except Exception as e:

        st.error(
            "Prediction error occurred."
        )

        st.code(str(e))
