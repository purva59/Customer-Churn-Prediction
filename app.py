import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SIMPLE PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #173b8f;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .section {
        font-size: 25px;
        font-weight: 750;
        color: #172554;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 18px;
        color: #475569;
    }

    .probability {
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
    }

    .small-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    .small-icon {
        font-size: 28px;
    }

    .small-title {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }

    .small-value {
        color: #172554;
        font-size: 21px;
        font-weight: 750;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD EXISTING XGBOOST MODEL
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


# Get feature names directly from your trained model
model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Customer Churn Analysis using XGBoost Classification</div>',
    unsafe_allow_html=True
)

st.info(
    "⚡ Powered by your trained XGBoost model"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Churn")

st.sidebar.write(
    "Enter customer information and generate a churn prediction."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model")

st.sidebar.write("Algorithm: XGBoost")

st.sidebar.write("Task: Classification")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Churn Prediction System"
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ============================================================
# CUSTOMER DETAILS
# ============================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("👤 Customer Details")

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
# INTERNET AND SERVICES
# ============================================================

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🌐 Internet & Services")

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
# BILLING
# ============================================================

st.markdown(
    '<div class="section">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
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
    '<div class="section">🔮 Generate Prediction</div>',
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

    # Create customer record
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


    # Convert categorical variables
    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # Match EXACTLY with training features
    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )


    # Convert to numeric
    customer_encoded = customer_encoded.astype(float)


    try:

        # ====================================================
        # ORIGINAL XGBOOST PREDICTION
        # ====================================================

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        probability_percent = probability * 100


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section">📌 Prediction Result</div>',
            unsafe_allow_html=True
        )


        if prediction == 1:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-title">⚠️ HIGH CHURN RISK</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-text">Customer is likely to CHURN</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="probability" style="color:#be123c;">'
                f'{probability_percent:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption("Churn Probability")

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "This customer may need retention offers or additional support."
            )


        else:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-title">✅ LOW CHURN RISK</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-text">Customer is likely to STAY</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="probability" style="color:#15803d;">'
                f'{probability_percent:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption("Churn Probability")

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.success(
                "The customer is predicted to continue the service."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section">📈 Prediction Summary</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)


        with m1:

            st.markdown(
                '<div class="small-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-icon">🎯</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-title">Prediction</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="small-value">'
                f'{"Churn" if prediction == 1 else "Stay"}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with m2:

            st.markdown(
                '<div class="small-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-icon">📊</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-title">Churn Probability</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="small-value">'
                f'{probability_percent:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with m3:

            st.markdown(
                '<div class="small-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-icon">⚡</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-title">Model</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="small-value">XGBoost</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # CHARTS
        # ====================================================

        st.markdown(
            '<div class="section">🎯 Churn Probability Analysis</div>',
            unsafe_allow_html=True
        )

        chart1, chart2 = st.columns(2)


        # ----------------------------------------------------
        # DONUT CHART
        # ----------------------------------------------------

        with chart1:

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Churn Probability",
                            "Remaining Probability"
                        ],
                        values=[
                            probability,
                            1 - probability
                        ],
                        hole=0.60,
                        textinfo="label+percent"
                    )
                ]
            )

            fig.update_layout(
                title="Churn Probability",
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

        with chart2:

            fig2 = go.Figure()

            fig2.add_trace(
                go.Bar(
                    x=["Churn"],
                    y=[probability_percent],
                    text=[f"{probability_percent:.2f}%"],
                    textposition="auto"
                )
            )

            fig2.update_layout(
                title="Churn Probability (%)",
                yaxis=dict(
                    range=[0, 100],
                    title="Probability"
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
        # RISK LEVEL
        # ====================================================

        st.markdown(
            '<div class="section">📍 Risk Level</div>',
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
        # CUSTOMER SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section">👤 Customer Input Summary</div>',
            unsafe_allow_html=True
        )

        summary1, summary2 = st.columns(2)


        with summary1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("Customer")

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


        with summary2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("Billing")

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

        st.error("❌ Prediction error occurred.")

        st.code(str(e))


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section">ℹ️ About This Model</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.markdown(
        '<div class="small-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-icon">🤖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-title">Algorithm</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-value">XGBoost</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        '<div class="small-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-icon">📋</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-title">Model Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="small-value">{len(model_features)}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        '<div class="small-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-icon">🎯</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-title">Task</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-value">Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with info4:

    st.markdown(
        '<div class="small-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-icon">📊</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-title">Output</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small-value">Churn / Stay</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Customer Churn Prediction System</b><br>
        Machine Learning Project using XGBoost
    </div>
    """,
    unsafe_allow_html=True
)
