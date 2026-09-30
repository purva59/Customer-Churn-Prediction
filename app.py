import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PROFESSIONAL LIGHT THEME
# ==========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #eaf3ff 0%, transparent 25%),
        radial-gradient(circle at 90% 5%, #f0ecff 0%, transparent 25%),
        #f7f9fc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6ebf2;
}

section[data-testid="stSidebar"] p {
    color: #63748a;
}

/* Headings */

h1 {
    color: #102a43 !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    color: #183b56 !important;
    font-weight: 750 !important;
}

h3 {
    color: #244e70 !important;
    font-weight: 700 !important;
}

/* Text */

p {
    color: #62748a;
}

/* Inputs */

div[data-baseweb="select"] > div {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
}

div[data-baseweb="input"] > div {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(100deg, #246bce, #6652d9);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 56px;
    font-size: 16px;
    font-weight: 800;
    box-shadow: 0 10px 25px rgba(45, 91, 170, 0.20);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 32px rgba(45, 91, 170, 0.30);
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e4eaf1;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(35, 65, 100, 0.06);
}

/* Expanders */

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 15px;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Divider */

hr {
    border-color: #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD EXISTING XGBOOST MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

model_features = model.get_booster().feature_names


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🤖 ChurnAI")

    st.caption("Customer Intelligence Platform")

    st.divider()

    st.markdown("### MODEL")

    st.write("**Algorithm**")
    st.write("XGBoost Classifier")

    st.write("**Task**")
    st.write("Binary Classification")

    st.write("**Target**")
    st.write("Customer Churn")

    st.divider()

    st.markdown("### ANALYSIS")

    st.write("👤 Customer Profile")
    st.write("🌐 Service Information")
    st.write("💳 Billing Information")
    st.write("🎯 Churn Prediction")
    st.write("📊 Risk Analysis")

    st.divider()

    st.success("Model Ready")

    st.caption(
        "AI-powered customer churn analysis"
    )


# ==========================================================
# HERO SECTION
# ==========================================================

hero1, hero2 = st.columns([3.5, 1])

with hero1:

    st.caption(
        "ARTIFICIAL INTELLIGENCE  •  CUSTOMER ANALYTICS"
    )

    st.title(
        "Customer Churn Prediction"
    )

    st.write(
        "Analyze customer information and estimate the "
        "probability of service churn using a trained "
        "XGBoost machine learning model."
    )

with hero2:

    st.metric(
        "AI MODEL",
        "XGBoost",
        "Ready"
    )


st.divider()


# ==========================================================
# DASHBOARD OVERVIEW
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Model",
        "XGBoost"
    )

with c2:
    st.metric(
        "Problem",
        "Classification"
    )

with c3:
    st.metric(
        "Output",
        "Churn Risk"
    )

with c4:
    st.metric(
        "Prediction",
        "Real-Time"
    )


st.write("")


# ==========================================================
# CUSTOMER PROFILE
# ==========================================================

st.header("Customer Information")

st.caption(
    "Enter the customer's details below."
)


with st.expander(
    "👤  CUSTOMER PROFILE",
    expanded=True
):

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

with st.expander(
    "📈  CUSTOMER VALUE",
    expanded=True
):

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

with st.expander(
    "🌐  SERVICE INFORMATION",
    expanded=True
):

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
# BILLING
# ==========================================================

with st.expander(
    "💳  BILLING & PAYMENT",
    expanded=True
):

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
# ANALYZE BUTTON
# ==========================================================

st.write("")

left, center, right = st.columns([1, 2, 1])

with center:

    analyze = st.button(
        "🔮  ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

if analyze:

    # ------------------------------------------------------
    # CUSTOMER DATA
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
    # SAME ENCODING
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
    # SAME MODEL PREDICTION
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

    st.caption(
        "AI ANALYSIS COMPLETE"
    )

    st.header(
        "Prediction Result"
    )


    result_col, gauge_col = st.columns(
        [1.2, 1]
    )


    # ------------------------------------------------------
    # RESULT CARD
    # ------------------------------------------------------

    with result_col:

        if prediction == 1:

            st.error(
                "⚠️ HIGHER CHURN RISK"
            )

            st.subheader(
                "Customer is predicted to be at higher risk of churn."
            )

        else:

            st.success(
                "✓ LOWER CHURN RISK"
            )

            st.subheader(
                "Customer is predicted to be at lower risk of churn."
            )

        st.write(
            "The result is generated by the trained "
            "XGBoost classification model."
        )

        st.write("")

        a, b = st.columns(2)

        with a:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%"
            )

        with b:

            st.metric(
                "Stay Probability",
                f"{stay_probability * 100:.1f}%"
            )


    # ------------------------------------------------------
    # GAUGE
    # ------------------------------------------------------

    with gauge_col:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=probability * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 38,
                        "color": "#183b56"
                    }
                },

                title={
                    "text": "CHURN RISK SCORE",
                    "font": {
                        "size": 15,
                        "color": "#52677d"
                    }
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#6257d9"
                    },

                    "bgcolor": "#eef2f7",

                    "borderwidth": 0,

                    "steps": [

                        {
                            "range": [0, 30],
                            "color": "#dff3e7"
                        },

                        {
                            "range": [30, 60],
                            "color": "#fff1c7"
                        },

                        {
                            "range": [60, 100],
                            "color": "#ffe1e1"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


    # ======================================================
    # PROBABILITY ANALYSIS
    # ======================================================

    st.subheader(
        "Probability Analysis"
    )

    chart_col, insight_col = st.columns(
        [1.5, 1]
    )


    with chart_col:

        chart = go.Figure()

        chart.add_trace(
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
                        "#3b82f6",
                        "#7659d9"
                    ]
                )
            )
        )

        chart.update_layout(

            height=360,

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            template="plotly_white",

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            margin=dict(
                l=40,
                r=20,
                t=30,
                b=40
            )
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )


    with insight_col:

        st.subheader(
            "Customer Risk Summary"
        )

        st.info(
            "The model evaluates customer profile, "
            "service usage, contract and billing information."
        )

        st.write("")

        st.write(
            f"**Churn:** {probability * 100:.1f}%"
        )

        st.progress(
            float(probability)
        )

        st.write(
            f"**Stay:** {stay_probability * 100:.1f}%"
        )

        st.progress(
            float(stay_probability)
        )

        st.write("")

        st.write(
            f"**Contract:** {contract}"
        )

        st.write(
            f"**Tenure:** {tenure} months"
        )

        st.write(
            f"**Internet:** {internet_service}"
        )


    # ======================================================
    # CUSTOMER SNAPSHOT
    # ======================================================

    st.divider()

    st.subheader(
        "Customer Snapshot"
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )

    with s2:

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.0f}"
        )

    with s3:

        st.metric(
            "Contract",
            contract
        )

    with s4:

        st.metric(
            "Internet",
            internet_service
        )


    # ======================================================
    # COMPLETE DETAILS
    # ======================================================

    with st.expander(
        "📋 View Complete Customer Information"
    ):

        details = pd.DataFrame({

            "Attribute": [

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
                tenure,
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
                monthly_charges,
                total_charges

            ]

        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "🤖 ChurnAI  •  Customer Churn Prediction  •  "
    "Powered by XGBoost"
)
