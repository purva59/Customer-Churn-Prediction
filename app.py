import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PAGE STYLE
# ==========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #f8f5ff 100%
        );
    }

    /* Main width */
    .block-container {
        max-width: 1350px;
        padding-top: 30px;
        padding-bottom: 40px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5eaf2;
    }

    /* Headings */
    h1 {
        color: #17365d !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #21466f !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #315579 !important;
        font-weight: 700 !important;
    }

    /* Input fields */
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border-color: #d9e3ef;
    }

    div[data-baseweb="input"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        border-color: #d9e3ef;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            90deg,
            #3978e8,
            #6c55e8
        );
        color: white;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(70, 90, 220, 0.20);
    }

    .stButton > button:hover {
        color: white;
        background: linear-gradient(
            90deg,
            #2868d8,
            #5d46d7
        );
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e1e8f0;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(40, 70, 110, 0.07);
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

# IMPORTANT:
# Get the exact features stored inside your trained model.
model_features = model.get_booster().feature_names


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🤖 ChurnAI")

    st.caption("Customer Churn Prediction System")

    st.divider()

    st.subheader("🧠 Machine Learning Model")

    st.write("**Algorithm**")
    st.write("XGBoost Classifier")

    st.write("**Problem Type**")
    st.write("Binary Classification")

    st.write("**Prediction**")
    st.write("Customer Churn")

    st.divider()

    st.subheader("📊 Analysis")

    st.write("👤 Customer Profile")
    st.write("📈 Customer Value")
    st.write("🌐 Services")
    st.write("💳 Billing")

    st.divider()

    st.info(
        "Enter customer details and use the trained "
        "XGBoost model to estimate churn probability."
    )


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title("🤖 Customer Churn Prediction")

st.write(
    "Analyze customer information and predict the probability "
    "of customer churn using an XGBoost machine learning model."
)


# ==========================================================
# CUSTOMER PROFILE
# ==========================================================

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


# ==========================================================
# CUSTOMER VALUE
# ==========================================================

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


# ==========================================================
# SERVICE INFORMATION
# ==========================================================

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


# ==========================================================
# BILLING INFORMATION
# ==========================================================

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


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict_button = st.button(
        "🔮  PREDICT CUSTOMER CHURN",
        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # ------------------------------------------------------
    # CREATE CUSTOMER DATA
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)


    # ------------------------------------------------------
    # XGBOOST PREDICTION
    # ------------------------------------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability


    # ======================================================
    # RESULT
    # ======================================================

    st.divider()

    st.header("🎯 AI Prediction")

    if prediction == 1:

        st.error(
            "⚠️ CUSTOMER AT RISK\n\n"
            "The model predicts a higher probability "
            "of customer churn."
        )

    else:

        st.success(
            "✅ CUSTOMER LIKELY TO STAY\n\n"
            "The model predicts a higher probability "
            "of customer retention."
        )


    # ======================================================
    # RESULT METRICS
    # ======================================================

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

        if prediction == 1:

            status = "Higher Risk"

        else:

            status = "Lower Risk"

        st.metric(
            "Risk Status",
            status
        )


    # ======================================================
    # PROBABILITY CHART
    # ======================================================

    st.subheader("📈 Prediction Analysis")

    chart1, chart2 = st.columns([1.4, 1])


    with chart1:

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
                marker_color=[
                    "#4F8FEF",
                    "#8B6FE8"
                ]
            )
        )

        fig.update_layout(

            title="Customer Churn Probability",

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            height=400,

            template="plotly_white",

            margin=dict(
                l=40,
                r=30,
                t=60,
                b=40
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with chart2:

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

        st.write(
            f"**Churn Probability:** "
            f"{probability * 100:.1f}%"
        )

        st.write(
            f"**Stay Probability:** "
            f"{stay_probability * 100:.1f}%"
        )

        if prediction == 1:

            st.warning(
                "The model has classified this customer "
                "as having a higher churn probability."
            )

        else:

            st.info(
                "The model has classified this customer "
                "as having a lower churn probability."
            )


    # ======================================================
    # CUSTOMER INFORMATION
    # ======================================================

    st.subheader("📋 Customer Information")

    with st.expander("View Customer Details"):

        customer_summary = pd.DataFrame({

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
            customer_summary,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🤖 ChurnAI | Customer Churn Analysis | "
    "XGBoost Machine Learning"
)
