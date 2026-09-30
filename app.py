import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f4f7fb;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #173b8f, #2563eb);
        padding: 35px 40px;
        border-radius: 22px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.20);
    }

    .main-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }

    .main-subtitle {
        color: #e8efff;
        font-size: 17px;
        text-align: center;
        margin-top: 8px;
    }

    .model-badge {
        text-align: center;
        margin-top: 18px;
    }

    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* Section title */
    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #172554;
        margin-top: 12px;
        margin-bottom: 15px;
    }

    /* Cards */
    .custom-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5eaf3;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    /* Small information cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5eaf3;
        text-align: center;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .info-icon {
        font-size: 27px;
    }

    .info-title {
        color: #64748b;
        font-size: 13px;
        margin-top: 5px;
    }

    .info-value {
        color: #172554;
        font-size: 21px;
        font-weight: 750;
        margin-top: 4px;
    }

    /* Result cards */
    .risk-high {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1px solid #fecdd3;
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
    }

    .risk-low {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #bbf7d0;
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
    }

    .risk-title {
        font-size: 30px;
        font-weight: 800;
    }

    .risk-high .risk-title {
        color: #be123c;
    }

    .risk-low .risk-title {
        color: #15803d;
    }

    .risk-probability {
        font-size: 23px;
        font-weight: 700;
        margin-top: 10px;
        color: #172554;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 25px 0 5px 0;
        margin-top: 30px;
        border-top: 1px solid #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5eaf3;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #173b8f, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 13px;
        font-size: 17px;
        font-weight: 700;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.25);
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #e5eaf3;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOUR EXISTING XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# Get feature names directly from your trained XGBoost model
model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">

    <div class="main-title">
        📊 Customer Churn Prediction
    </div>

    <div class="main-subtitle">
        Predict whether a customer is likely to discontinue the service
        using the trained XGBoost classification model.
    </div>

    <div class="model-badge">
        <span class="badge">
            ⚡ Powered by XGBoost
        </span>
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 📊 Churn Predictor")

st.sidebar.markdown(
    """
    <div style="
        background:#f4f7fb;
        padding:15px;
        border-radius:12px;
        margin-bottom:20px;
    ">
        <b>Customer Churn Analysis</b><br>
        <span style="color:#64748b;font-size:13px;">
        Enter customer information and generate a churn prediction.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 📌 Model")

st.sidebar.info(
    "XGBoost Classification Model"
)

st.sidebar.markdown("### 📋 Input Information")

st.sidebar.write(
    "Customer, service, contract and billing details are used for prediction."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Churn Prediction System"
)


# ============================================================
# CUSTOMER DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ============================================================
# CUSTOMER DETAILS CARD
# ============================================================

with col1:

    st.markdown(
        '<div class="custom-card">',
        unsafe_allow_html=True
    )

    st.markdown("### 👤 Customer Details")

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

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# INTERNET & SERVICES
# ============================================================

with col2:

    st.markdown(
        '<div class="custom-card">',
        unsafe_allow_html=True
    )

    st.markdown("### 🌐 Internet & Services")

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

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="custom-card">',
    unsafe_allow_html=True
)

bill1, bill2, bill3 = st.columns(3)


with bill1:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with bill2:

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with bill3:

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=800.0
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Generate Prediction</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create customer record
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
    # Convert categorical variables
    # --------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # --------------------------------------------------------
    # Match EXACTLY with training features
    # --------------------------------------------------------

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )


    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    customer_encoded = customer_encoded.astype(float)


    try:

        # ----------------------------------------------------
        # YOUR ORIGINAL MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]


        probability_percent = probability * 100


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📌 Prediction Result</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # HIGH CHURN
        # ====================================================

        if prediction == 1:

            st.markdown(
                f"""
                <div class="risk-high">

                    <div class="risk-title">
                        ⚠️ HIGH CHURN RISK
                    </div>

                    <div class="risk-probability">
                        Customer is likely to CHURN
                    </div>

                    <div style="
                        margin-top:15px;
                        color:#475569;
                        font-size:16px;
                    ">
                        Churn Probability
                    </div>

                    <div style="
                        font-size:38px;
                        font-weight:800;
                        color:#be123c;
                        margin-top:5px;
                    ">
                        {probability_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "⚠️ This customer may need retention offers, "
                "additional support, or engagement strategies."
            )


        # ====================================================
        # LOW CHURN
        # ====================================================

        else:

            st.markdown(
                f"""
                <div class="risk-low">

                    <div class="risk-title">
                        ✅ LOW CHURN RISK
                    </div>

                    <div class="risk-probability">
                        Customer is likely to STAY
                    </div>

                    <div style="
                        margin-top:15px;
                        color:#475569;
                        font-size:16px;
                    ">
                        Churn Probability
                    </div>

                    <div style="
                        font-size:38px;
                        font-weight:800;
                        color:#15803d;
                        margin-top:5px;
                    ">
                        {probability_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "✅ The customer is predicted to continue the service."
            )


        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        st.markdown(
            '<div class="section-title">📈 Prediction Summary</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)


        with m1:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-icon">🎯</div>

                    <div class="info-title">
                        Prediction
                    </div>

                    <div class="info-value">
                        {"Churn" if prediction == 1 else "Stay"}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with m2:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-icon">📊</div>

                    <div class="info-title">
                        Churn Probability
                    </div>

                    <div class="info-value">
                        {probability_percent:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with m3:

            st.markdown(
                """
                <div class="info-card">

                    <div class="info-icon">⚡</div>

                    <div class="info-title">
                        Model
                    </div>

                    <div class="info-value">
                        XGBoost
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY VISUALIZATION
        # ====================================================

        st.markdown(
            '<div class="section-title">🎯 Churn Probability Analysis</div>',
            unsafe_allow_html=True
        )

        chart_col1, chart_col2 = st.columns(2)


        # ----------------------------------------------------
        # DONUT CHART
        # ----------------------------------------------------

        with chart_col1:

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Churn Probability", "Remaining"],
                        values=[
                            probability,
                            1 - probability
                        ],
                        hole=0.65,
                        textinfo="none"
                    )
                ]
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                ),
                showlegend=True,
                title={
                    "text": "Churn Probability",
                    "x": 0.5
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

        with chart_col2:

            fig2 = go.Figure()

            fig2.add_trace(
                go.Bar(
                    x=["Churn Probability"],
                    y=[probability_percent],
                    text=[f"{probability_percent:.2f}%"],
                    textposition="auto"
                )
            )

            fig2.update_layout(
                title={
                    "text": "Prediction Confidence",
                    "x": 0.5
                },
                yaxis=dict(
                    title="Probability (%)",
                    range=[0, 100]
                ),
                height=350,
                margin=dict(
                    l=50,
                    r=20,
                    t=60,
                    b=40
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


        # ====================================================
        # PROGRESS INDICATOR
        # ====================================================

        st.markdown(
            '<div class="section-title">📍 Risk Level</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(probability)
        )

        if probability >= 0.70:

            st.error(
                f"High probability of churn: "
                f"{probability_percent:.2f}%"
            )

        elif probability >= 0.40:

            st.warning(
                f"Moderate probability of churn: "
                f"{probability_percent:.2f}%"
            )

        else:

            st.success(
                f"Lower probability of churn: "
                f"{probability_percent:.2f}%"
            )


        # ====================================================
        # CUSTOMER INPUT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">👤 Customer Input Summary</div>',
            unsafe_allow_html=True
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.markdown(
                '<div class="custom-card">',
                unsafe_allow_html=True
            )

            st.markdown("### Customer")

            st.write(f"**Gender:** {gender}")
            st.write(f"**Senior Citizen:** {senior}")
            st.write(f"**Partner:** {partner}")
            st.write(f"**Dependents:** {dependents}")
            st.write(f"**Tenure:** {tenure} months")
            st.write(f"**Phone Service:** {phone}")

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with summary_col2:

            st.markdown(
                '<div class="custom-card">',
                unsafe_allow_html=True
            )

            st.markdown("### Billing")

            st.write(f"**Contract:** {contract}")
            st.write(f"**Payment Method:** {payment}")
            st.write(f"**Monthly Charges:** ${monthly:.2f}")
            st.write(f"**Total Charges:** ${total:.2f}")
            st.write(f"**Paperless Billing:** {paperless}")

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction error occurred."
        )

        st.code(str(e))


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">ℹ️ About This Model</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">🤖</div>

            <div class="info-title">
                Algorithm
            </div>

            <div class="info-value">
                XGBoost
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-icon">📋</div>

            <div class="info-title">
                Model Features
            </div>

            <div class="info-value">
                {len(model_features)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">🎯</div>

            <div class="info-title">
                Task
            </div>

            <div class="info-value">
                Classification
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info4:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">📊</div>

            <div class="info-title">
                Output
            </div>

            <div class="info-value">
                Churn / Stay
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>Customer Churn Prediction System</b><br>

        Machine Learning Project using XGBoost<br>

        Customer Churn Analysis & Prediction

    </div>
    """,
    unsafe_allow_html=True
)
