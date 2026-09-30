import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #1f3c88;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

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


# Get feature names directly from trained XGBoost model
model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">XGBoost Based Customer Churn Analysis System</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Customer Information")

st.sidebar.write(
    "Enter customer details and click **Predict Churn**."
)


# ---------------- INPUTS ----------------

col1, col2 = st.columns(2)


with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)


with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- BILLING ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("💳 Contract & Billing")

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

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- PREDICTION BUTTON ----------------

st.markdown("### 🔮 Prediction")

predict_button = st.button(
    "🚀 Predict Customer Churn",
    use_container_width=True
)


# ---------------- PREDICTION ----------------

if predict_button:

    # Create original-style customer record

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


    # Convert data to numeric
    customer_encoded = customer_encoded.astype(float)


    try:

        prediction = model.predict(customer_encoded)[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]


        probability_percent = probability * 100


        # ---------------- RESULT ----------------

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ HIGH CHURN RISK"
            )

            st.markdown(
                f"""
                <div class="result">

                ⚠️ Customer is likely to CHURN

                <br><br>

                Churn Probability:
                {probability_percent:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "This customer may need retention offers or additional support."
            )


        else:

            st.success(
                "✅ LOW CHURN RISK"
            )

            st.markdown(
                f"""
                <div class="result">

                ✅ Customer is likely to STAY

                <br><br>

                Churn Probability:
                {probability_percent:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "The customer is predicted to continue the service."
            )


        # ---------------- METRICS ----------------

        st.markdown("### 📈 Prediction Summary")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Prediction",
            "Churn" if prediction == 1 else "Stay"
        )

        m2.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        m3.metric(
            "Model",
            "XGBoost"
        )


        # ---------------- PROBABILITY BAR ----------------

        st.markdown("### 🎯 Churn Probability")

        st.progress(
            float(probability)
        )


    except Exception as e:

        st.error(
            "Prediction error occurred."
        )

        st.code(str(e))