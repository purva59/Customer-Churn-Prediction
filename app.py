
import streamlit as st
import pandas as pd
import numpy as np
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
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f4f8ff 0%,
        #eef4ff 50%,
        #f8fbff 100%
    );
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6ebf2;
}

[data-testid="stSidebar"] h1 {
    color: #17365d;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 17px;
    font-weight: 600;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 750;
    color: #17365d;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 17px;
    color: #667085;
    margin-bottom: 28px;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e8edf5;
    box-shadow: 0 5px 20px rgba(23, 54, 93, 0.07);
    margin-bottom: 20px;
}

/* Section title */
.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #17365d;
    margin-bottom: 12px;
}

/* Result */
.result-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #e5eaf2;
    box-shadow: 0 5px 20px rgba(23, 54, 93, 0.08);
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 30px;
    font-weight: 750;
    margin-bottom: 10px;
}

.probability {
    font-size: 36px;
    font-weight: 750;
    color: #17365d;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 15px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 45px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #e8edf5;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD EXACT EXISTING MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


try:
    model = load_model()

except Exception as e:

    st.error("❌ churn_model.json could not be loaded.")

    st.write("Make sure that churn_model.json is in the same folder as app.py.")

    st.code(str(e))

    st.stop()


# ============================================================
# GET EXACT FEATURES FROM TRAINED MODEL
# ============================================================

model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "❌ The trained model does not contain feature names."
    )

    st.stop()


# ============================================================
# ORIGINAL RAW CUSTOMER FEATURES
# These are NOT changed.
# ============================================================

raw_features = [
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


# ============================================================
# HELPER FUNCTION
# ============================================================

def prepare_customer_data(customer_df):
    """
    Converts original customer data into the EXACT
    feature format expected by churn_model.json.
    """

    data = customer_df.copy()

    # Remove customerID if present
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    # Remove actual Churn column if uploaded CSV contains it
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

    # Convert TotalCharges to numeric
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    # Convert categorical variables exactly as original model
    data_encoded = pd.get_dummies(
        data,
        drop_first=True
    )

    # Match EXACT trained model columns
    data_encoded = data_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # XGBoost requires numeric values
    data_encoded = data_encoded.astype(float)

    return data_encoded


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'XGBoost-based Customer Churn Analysis & Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

st.sidebar.markdown(
    "Select an option below:"
)

page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "🎯 Churn Prediction",
        "🤖 Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: XGBoost\n\n"
    "File: churn_model.json\n\n"
    "Mode: Classification"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Welcome to Customer Churn Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application uses the trained XGBoost model to "
        "identify whether a customer is likely to churn or stay."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # Dashboard metrics

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Model",
            "XGBoost"
        )

    with c2:
        st.metric(
            "Prediction Type",
            "Binary Classification"
        )

    with c3:
        st.metric(
            "Output",
            "Churn / Stay"
        )

    st.markdown("### 🔍 Available Features")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("#### 👤 Manual Prediction")

        st.write(
            "Enter customer details manually and get "
            "the churn prediction and probability."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("#### 📂 CSV File Prediction")

        st.write(
            "Upload a customer CSV, preview the data, "
            "predict churn for all customers and download results."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🎯 Churn Prediction":

    st.subheader("🎯 Churn Prediction")

    prediction_type = st.radio(
        "Choose Prediction Method",
        [
            "👤 Manual Customer Prediction",
            "📂 CSV File Prediction"
        ],
        horizontal=True
    )


    # ========================================================
    # MANUAL CUSTOMER PREDICTION
    # ========================================================

    if prediction_type == "👤 Manual Customer Prediction":

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">👤 Customer Details</div>',
            unsafe_allow_html=True
        )

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

        with col2:

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with col3:

            protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            support = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # BILLING
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">💳 Contract & Billing</div>',
            unsafe_allow_html=True
        )

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


        # ====================================================
        # PREDICTION BUTTON
        # ====================================================

        predict_button = st.button(
            "🔮 Predict Customer Churn",
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

                customer_encoded = prepare_customer_data(
                    customer
                )

                prediction = model.predict(
                    customer_encoded
                )[0]

                probability = model.predict_proba(
                    customer_encoded
                )[0][1]

                probability_percent = probability * 100


                st.markdown("---")

                # Result

                if prediction == 1:

                    st.error(
                        "⚠️ HIGH CHURN RISK"
                    )

                    result_text = "Customer is predicted to CHURN"

                else:

                    st.success(
                        "✅ LOW CHURN RISK"
                    )

                    result_text = "Customer is predicted to STAY"


                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            {result_text}
                        </div>

                        <div>
                            Churn Probability
                        </div>

                        <div class="probability">
                            {probability_percent:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown("### 📈 Prediction Summary")

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


                st.markdown("### 🎯 Churn Probability")

                st.progress(
                    float(probability)
                )


            except Exception as e:

                st.error(
                    "❌ Prediction error occurred."
                )

                st.code(
                    str(e)
                )


    # ========================================================
    # CSV FILE PREDICTION
    # ========================================================

    else:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">📂 CSV File Prediction</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Upload a customer CSV file. The application will "
            "preview the data and use the existing "
            "**churn_model.json** for prediction."
        )

        st.info(
            "The uploaded CSV should use the same original "
            "customer features used by the trained model. "
            "The Churn column is optional."
        )

        uploaded_file = st.file_uploader(
            "Choose Customer CSV",
            type=["csv"],
            key="customer_csv"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        if uploaded_file is not None:

            try:

                csv_data = pd.read_csv(
                    uploaded_file
                )

                # --------------------------------------------
                # PREVIEW
                # --------------------------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">👀 CSV Preview</div>',
                    unsafe_allow_html=True
                )

                r1, r2, r3 = st.columns(3)

                with r1:
                    st.metric(
                        "Rows",
                        csv_data.shape[0]
                    )

                with r2:
                    st.metric(
                        "Columns",
                        csv_data.shape[1]
                    )

                with r3:
                    st.metric(
                        "Model Features",
                        len(model_features)
                    )

                st.dataframe(
                    csv_data.head(10),
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

                if st.button(
                    "🔮 Predict CSV",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing customer data..."
                    ):

                        prepared_data = prepare_customer_data(
                            csv_data
                        )

                        predictions = model.predict(
                            prepared_data
                        )

                        probabilities = model.predict_proba(
                            prepared_data
                        )[:, 1]


                    # ----------------------------------------
                    # RESULTS
                    # ----------------------------------------

                    result_data = csv_data.copy()

                    result_data["Prediction"] = np.where(
                        predictions == 1,
                        "Churn",
                        "Stay"
                    )

                    result_data["Churn Probability"] = (
                        probabilities * 100
                    ).round(2)


                    churn_count = int(
                        (predictions == 1).sum()
                    )

                    stay_count = int(
                        (predictions == 0).sum()
                    )

                    total_customers = len(
                        result_data
                    )


                    st.success(
                        "✅ CSV prediction completed successfully!"
                    )


                    # ----------------------------------------
                    # SUMMARY
                    # ----------------------------------------

                    st.markdown(
                        "### 📊 Prediction Summary"
                    )

                    s1, s2, s3 = st.columns(3)

                    with s1:

                        st.metric(
                            "Total Customers",
                            total_customers
                        )

                    with s2:

                        st.metric(
                            "Predicted Churn",
                            churn_count
                        )

                    with s3:

                        st.metric(
                            "Predicted Stay",
                            stay_count
                        )


                    # ----------------------------------------
                    # RESULTS TABLE
                    # ----------------------------------------

                    st.markdown(
                        "### 📋 Prediction Results"
                    )

                    st.dataframe(
                        result_data,
                        use_container_width=True
                    )


                    # ----------------------------------------
                    # DOWNLOAD
                    # ----------------------------------------

                    output_csv = result_data.to_csv(
                        index=False
                    ).encode("utf-8")


                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=output_csv,
                        file_name="customer_churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


            except Exception as e:

                st.error(
                    "❌ CSV prediction could not be completed."
                )

                st.write(
                    "Please make sure the uploaded CSV contains "
                    "the customer features expected by the model."
                )

                st.code(
                    str(e)
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.subheader("🤖 Model Information")

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### XGBoost Customer Churn Model"
    )

    st.write(
        "The application uses the existing trained "
        "`churn_model.json` model."
    )

    st.write(
        "**No retraining is performed inside this application.**"
    )

    st.write(
        "**No changes are made to the trained model features.**"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # Model details

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Algorithm",
            "XGBoost"
        )

    with c2:

        st.metric(
            "Model File",
            "churn_model.json"
        )

    with c3:

        st.metric(
            "Features",
            len(model_features)
        )


    st.markdown("### 🔑 Exact Model Features")

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
        "The feature list above is read directly from "
        "the trained XGBoost model."
    )
```
