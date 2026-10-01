import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
# PROFESSIONAL LIGHT UI
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f9ff 0%, #eef4ff 50%, #f8fbff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #eef5ff 100%);
        border-right: 1px solid #dbe5f1;
    }

    [data-testid="stSidebar"] * {
        font-size: 18px !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
    }

    h1 {
        color: #17365d;
        font-weight: 800;
    }

    h2, h3 {
        color: #234e7d;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce6f2;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(40, 80, 120, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("churn_model.json")
    return model


model = load_model()

# Get exact feature names from the existing model
model_features = model.get_booster().feature_names


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("📊 Customer Churn")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🎯 Churn Prediction",
        "🤖 Model Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "XGBoost based Customer Churn Prediction System"
)


# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.title("📊 Customer Churn Analysis")
    st.subheader("XGBoost Based Customer Churn Prediction")

    st.write(
        "This application predicts whether a customer is likely to "
        "discontinue the service using an existing XGBoost classification model."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 Algorithm",
            "XGBoost"
        )

    with col2:
        st.metric(
            "🎯 Task",
            "Binary Classification"
        )

    with col3:
        st.metric(
            "📌 Output",
            "Churn / Stay"
        )

    st.markdown("---")

    st.subheader("🔍 What This System Does")

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            "👤 **Manual Prediction**\n\n"
            "Enter customer information manually and get "
            "the predicted churn status and probability."
        )

    with c2:
        st.info(
            "📂 **CSV Prediction**\n\n"
            "Upload multiple customer records and generate "
            "churn predictions for the complete file."
        )

    st.markdown("---")

    st.subheader("📈 Prediction Flow")

    st.write(
        "Customer Data → Data Preprocessing → XGBoost Model → "
        "Churn Probability → Churn / Stay Prediction"
    )


# =========================================================
# CHURN PREDICTION
# =========================================================
elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")

    st.write(
        "Choose a prediction method below."
    )

    manual_tab, csv_tab = st.tabs(
        ["👤 Manual Customer Prediction", "📂 CSV File Prediction"]
    )

    # =====================================================
    # MANUAL PREDICTION
    # =====================================================
    with manual_tab:

        st.subheader("👤 Enter Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
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
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        with col2:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=1000.0
            )

        st.markdown("---")

        predict_button = st.button(
            "🔮 Predict Customer Churn",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            customer = pd.DataFrame({
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
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

            # Same preprocessing used for the existing model
            customer_encoded = pd.get_dummies(
                customer,
                drop_first=True
            )

            customer_encoded = customer_encoded.reindex(
                columns=model_features,
                fill_value=0
            )

            customer_encoded = customer_encoded.astype(float)

            # Prediction
            prediction = model.predict(
                customer_encoded
            )[0]

            probability = model.predict_proba(
                customer_encoded
            )[0][1]

            probability_percent = probability * 100

            st.markdown("---")

            st.subheader("📊 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:
                    st.error("⚠️ Customer is likely to CHURN")
                else:
                    st.success("✅ Customer is likely to STAY")

            with result_col2:

                st.metric(
                    "Churn Probability",
                    f"{probability_percent:.2f}%"
                )

            st.progress(
                float(probability)
            )

            if probability >= 0.5:
                st.warning(
                    "The predicted churn probability is above the "
                    "classification threshold."
                )
            else:
                st.success(
                    "The predicted churn probability is below the "
                    "classification threshold."
                )

            # Gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percent,
                    title={
                        "text": "Churn Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": 50
                        }
                    }
                )
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # =====================================================
    # CSV BATCH PREDICTION
    # =====================================================
    with csv_tab:

        st.subheader("📂 Upload Customer CSV File")

        st.write(
            "Upload a CSV file containing customer information. "
            "The application will generate churn predictions for all records."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    "✅ CSV file uploaded successfully."
                )

                st.subheader("👀 Data Preview")

                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True
                )

                st.write(
                    f"Total records: **{len(uploaded_df)}**"
                )

                # Required columns
                required_columns = [
                    "gender",
                    "SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "tenure",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                    "MonthlyCharges",
                    "TotalCharges"
                ]

                missing_columns = [
                    col
                    for col in required_columns
                    if col not in uploaded_df.columns
                ]

                if missing_columns:

                    st.error(
                        "❌ Required columns are missing:"
                    )

                    st.write(
                        missing_columns
                    )

                else:

                    st.markdown("---")

                    predict_csv = st.button(
                        "🔮 Predict CSV File",
                        type="primary",
                        use_container_width=True
                    )

                    if predict_csv:

                        # Keep original data for output
                        result_df = uploaded_df.copy()

                        # Create model input
                        prediction_df = uploaded_df.copy()

                        # Remove columns that are not model inputs
                        if "customerID" in prediction_df.columns:
                            prediction_df = prediction_df.drop(
                                "customerID",
                                axis=1
                            )

                        # Remove actual Churn column if present
                        if "Churn" in prediction_df.columns:
                            prediction_df = prediction_df.drop(
                                "Churn",
                                axis=1
                            )

                        # Convert TotalCharges to numeric
                        prediction_df["TotalCharges"] = pd.to_numeric(
                            prediction_df["TotalCharges"],
                            errors="coerce"
                        )

                        # Fill missing TotalCharges
                        prediction_df["TotalCharges"] = (
                            prediction_df["TotalCharges"]
                            .fillna(
                                prediction_df["TotalCharges"].median()
                            )
                        )

                        # One-hot encoding
                        encoded_df = pd.get_dummies(
                            prediction_df,
                            drop_first=True
                        )

                        # Match exact model features
                        encoded_df = encoded_df.reindex(
                            columns=model_features,
                            fill_value=0
                        )

                        encoded_df = encoded_df.astype(float)

                        # Predictions
                        predictions = model.predict(
                            encoded_df
                        )

                        probabilities = model.predict_proba(
                            encoded_df
                        )[:, 1]

                        # Add results
                        result_df["Predicted_Churn"] = [
                            "Churn" if value == 1 else "Stay"
                            for value in predictions
                        ]

                        result_df[
                            "Churn_Probability_Percent"
                        ] = (
                            probabilities * 100
                        ).round(2)

                        st.success(
                            "✅ Predictions generated successfully!"
                        )

                        st.markdown("---")

                        st.subheader("📊 Prediction Summary")

                        total_customers = len(result_df)

                        churn_count = (
                            result_df["Predicted_Churn"]
                            .eq("Churn")
                            .sum()
                        )

                        stay_count = (
                            result_df["Predicted_Churn"]
                            .eq("Stay")
                            .sum()
                        )

                        average_probability = (
                            result_df[
                                "Churn_Probability_Percent"
                            ].mean()
                        )

                        m1, m2, m3, m4 = st.columns(4)

                        with m1:
                            st.metric(
                                "👥 Total Customers",
                                total_customers
                            )

                        with m2:
                            st.metric(
                                "⚠️ Predicted Churn",
                                churn_count
                            )

                        with m3:
                            st.metric(
                                "✅ Predicted Stay",
                                stay_count
                            )

                        with m4:
                            st.metric(
                                "📈 Avg. Churn Probability",
                                f"{average_probability:.2f}%"
                            )

                        st.markdown("---")

                        st.subheader("📋 Prediction Results")

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        # Download
                        csv_bytes = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="⬇️ Download Prediction Results",
                            data=csv_bytes,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:

                st.error(
                    "❌ Error while processing the CSV file."
                )

                st.code(
                    str(e)
                )


# =========================================================
# MODEL INFORMATION
# =========================================================
elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.subheader("XGBoost Classification Model")

    st.write(
        "The application uses an existing XGBoost classification "
        "model to predict customer churn."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "### Algorithm\n"
            "XGBoost Classifier"
        )

        st.info(
            "### Problem Type\n"
            "Binary Classification"
        )

        st.info(
            "### Target\n"
            "Customer Churn"
        )

    with col2:

        st.info(
            "### Prediction 1\n"
            "Churn"
        )

        st.info(
            "### Prediction 2\n"
            "Stay"
        )

        st.info(
            "### Model File\n"
            "churn_model.json"
        )

    st.markdown("---")

    st.subheader("🔧 Model Configuration")

    config_df = pd.DataFrame({
        "Parameter": [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "eval_metric"
        ],
        "Value": [
            "100",
            "4",
            "0.1",
            "logloss"
        ]
    })

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📌 Input Features")

    st.write(
        f"The model uses **{len(model_features)} encoded features** "
        "after preprocessing."
    )

    with st.expander("View Model Features"):

        st.write(
            model_features
        )

    st.markdown("---")

    st.info(
        "This application uses the existing trained model. "
        "No retraining is performed inside the application."
    )
