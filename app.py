import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# =========================================================

# PAGE CONFIGURATION

# =========================================================

st.set_page_config(
page_title="Customer Churn Prediction",
page_icon="📊",
layout="wide"
)

# =========================================================

# CUSTOM CSS

# =========================================================

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
    color: #555555;
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
    font-size: 24px;
    font-weight: bold;
}
</style>

""", unsafe_allow_html=True)

# =========================================================

# LOAD EXISTING MODEL

# =========================================================

@st.cache_resource
def load_model():
model = XGBClassifier()
model.load_model("churn_model.json")
return model

try:
model = load_model()
model_features = model.get_booster().feature_names

```
if model_features is None:
    st.error("Model feature names are not available.")
    st.stop()
```

except Exception as e:
st.error("❌ churn_model.json could not be loaded.")
st.code(str(e))
st.stop()

# =========================================================

# FUNCTION TO PREPARE DATA

# =========================================================

def prepare_data(data):

```
data = data.copy()

# Remove ID and target if present
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

if "Churn" in data.columns:
    data = data.drop("Churn", axis=1)

# Convert TotalCharges
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

# Fill missing numeric values
if "TotalCharges" in data.columns:
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

# One-hot encoding
data = pd.get_dummies(
    data,
    drop_first=True
)

# Match exactly with model features
data = data.reindex(
    columns=model_features,
    fill_value=0
)

# Convert to numeric
data = data.astype(float)

return data
```

# =========================================================

# HEADER

# =========================================================

st.markdown(
'<div class="title">📊 Customer Churn Prediction</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">XGBoost Based Customer Churn Analysis System</div>',
unsafe_allow_html=True
)

# =========================================================

# SIDEBAR NAVIGATION

# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
"Select Page",
[
"🏠 Dashboard",
"🎯 Churn Prediction",
"🤖 Model Information"
]
)

# =========================================================

# DASHBOARD

# =========================================================

if page == "🏠 Dashboard":

```
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🏠 Customer Churn Dashboard")

st.write(
    """
    This application predicts whether a customer is likely to
    discontinue the service using an existing XGBoost classification model.
    """
)

st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🤖 Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "📊 Prediction Type",
        "Classification"
    )

with col3:
    st.metric(
        "📁 Input Methods",
        "Manual + CSV"
    )

st.markdown("---")

st.subheader("📌 Available Features")

f1, f2 = st.columns(2)

with f1:
    st.info(
        """
        👤 Manual Customer Prediction

        Enter customer information manually and get:
        - Churn prediction
        - Churn probability
        """
    )

with f2:
    st.info(
        """
        📂 CSV File Prediction

        Upload multiple customer records and get:
        - Prediction for each customer
        - Churn probability
        - Downloadable results
        """
    )
```

# =========================================================

# CHURN PREDICTION

# =========================================================

elif page == "🎯 Churn Prediction":

```
st.header("🎯 Churn Prediction")

tab1, tab2 = st.tabs(
    [
        "👤 Manual Customer Prediction",
        "📂 CSV File Prediction"
    ]
)

# =====================================================
# MANUAL PREDICTION
# =====================================================

with tab1:

    st.subheader("👤 Customer Details")

    col1, col2 = st.columns(2)

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

    with col2:

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

    st.subheader("💳 Contract & Billing")

    b1, b2, b3 = st.columns(3)

    with b1:

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

    with b2:

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with b3:

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

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Customer Churn",
        use_container_width=True
    )

    if predict_button:

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

        try:

            customer_encoded = prepare_data(customer)

            prediction = model.predict(
                customer_encoded
            )[0]

            probability = model.predict_proba(
                customer_encoded
            )[0][1]

            probability_percent = probability * 100

            st.markdown("---")

            if prediction == 1:

                st.error("⚠️ HIGH CHURN RISK")

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

            else:

                st.success("✅ LOW CHURN RISK")

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

            st.subheader("📈 Prediction Summary")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "Stay"
                )

            with m2:
                st.metric(
                    "Churn Probability",
                    f"{probability_percent:.2f}%"
                )

            with m3:
                st.metric(
                    "Model",
                    "XGBoost"
                )

            st.subheader("🎯 Churn Probability")

            st.progress(float(probability))

        except Exception as e:

            st.error("❌ Prediction error")
            st.code(str(e))


# =====================================================
# CSV PREDICTION
# =====================================================

with tab2:

    st.subheader("📂 Upload Customer CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.success("✅ CSV file uploaded successfully!")

            st.subheader("👀 Data Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.write(
                f"Total Records: **{len(df)}**"
            )

            if st.button(
                "🚀 Predict CSV Customers",
                use_container_width=True
            ):

                input_data = prepare_data(df)

                predictions = model.predict(
                    input_data
                )

                probabilities = model.predict_proba(
                    input_data
                )[:, 1]

                result_df = df.copy()

                result_df["Prediction"] = [
                    "Churn" if x == 1 else "Stay"
                    for x in predictions
                ]

                result_df["Churn Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                st.success(
                    "✅ Prediction completed successfully!"
                )

                st.subheader("📊 Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                churn_count = int(
                    (predictions == 1).sum()
                )

                stay_count = int(
                    (predictions == 0).sum()
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Customers",
                        len(result_df)
                    )

                with c2:
                    st.metric(
                        "Predicted Churn",
                        churn_count
                    )

                with c3:
                    st.metric(
                        "Predicted Stay",
                        stay_count
                    )

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "❌ Error while processing CSV file"
            )

            st.code(str(e))
```

# =========================================================

# MODEL INFORMATION

# =========================================================

elif page == "🤖 Model Information":

```
st.header("🤖 Model Information")

st.markdown(
    """
    ### XGBoost Classification Model

    This application uses a pre-trained **XGBoost classification
    model** stored in `churn_model.json`.

    The model is loaded directly from the existing JSON file.
    No retraining is performed by this application.
    """
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("📌 Model Details")

    st.write("**Algorithm:** XGBoost")
    st.write("**Task:** Binary Classification")
    st.write("**Output:** Churn / Stay")
    st.write("**Probability:** Churn Probability")

with col2:

    st.subheader("📊 Model Features")

    st.write(
        f"Number of features: **{len(model_features)}**"
    )

st.markdown("---")

st.subheader("🔢 Features Used by Model")

feature_df = pd.DataFrame(
    {
        "Feature": model_features
    }
)

st.dataframe(
    feature_df,
    use_container_width=True
)

st.info(
    "The feature list is read directly from the existing trained XGBoost model."
)
```
