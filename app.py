import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef6ff, #f8fbff);
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    .title {
        font-size: 38px;
        font-weight: 700;
        color: #17365d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6b7a;
        margin-bottom: 25px;
    }

    .card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .result {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: white;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    }

    .big-number {
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">XGBoost-based customer churn analysis and prediction system</div>',
    unsafe_allow_html=True
)


# =========================================================
# FUNCTIONS
# =========================================================

MODEL_FILE = "churn_model.pkl"


def clean_data(df):
    """Clean customer churn dataset."""

    df = df.copy()

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove customerID because it is only an identifier
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df


def train_model(df):
    """Train XGBoost model using uploaded training CSV."""

    df = clean_data(df)

    # Check target
    if "Churn" not in df.columns:
        st.error("❌ Training CSV must contain a 'Churn' column.")
        return None, None

    # Remove rows where target is missing
    df = df.dropna(subset=["Churn"])

    # Convert target
    if df["Churn"].dtype == "object":
        df["Churn"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0
            })
        )

    # Remove invalid target rows
    df = df.dropna(subset=["Churn"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)

    # Store feature names
    feature_columns = X.columns.tolist()

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="drop"
    )

    # XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Test
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred
        ),
        "features": feature_columns
    }

    # Save model and feature information
    model_package = {
        "model": pipeline,
        "features": feature_columns
    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    return model_package, metrics


def load_saved_model():
    """Load saved model."""

    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    return None


def prepare_prediction_data(df, training_features):
    """
    Prepare uploaded CSV for prediction.

    Missing training columns are added with NaN.
    Extra columns are ignored.
    """

    df = df.copy()

    df.columns = df.columns.str.strip()

    # Remove target if present
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    # Remove customer ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Add missing columns
    for column in training_features:
        if column not in df.columns:
            df[column] = np.nan

    # Keep only training columns
    df = df[training_features]

    return df


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🤖 Train Model",
        "📂 CSV Prediction",
        "👤 Manual Prediction",
        "ℹ️ Model Information"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.subheader("Welcome to Customer Churn Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "Prediction Type",
            "Binary Classification"
        )

    with col3:
        st.metric(
            "Target",
            "Customer Churn"
        )

    st.markdown("---")

    st.info(
        """
        This application predicts whether a customer is likely to
        **Churn** or **Stay** based on customer information.

        You can:
        - Train the model using a CSV file
        - Upload another compatible CSV
        - Predict churn for multiple customers
        - Download prediction results
        - Predict one customer manually
        """
    )


# =========================================================
# TRAIN MODEL
# =========================================================

elif page == "🤖 Train Model":

    st.subheader("🤖 Train XGBoost Model")

    st.write(
        "Upload your training CSV file. The CSV must contain a **Churn** column."
    )

    uploaded_train = st.file_uploader(
        "Upload Training CSV",
        type=["csv"],
        key="training_file"
    )

    if uploaded_train is not None:

        train_df = pd.read_csv(uploaded_train)

        st.success(
            f"CSV loaded successfully: {train_df.shape[0]} rows, "
            f"{train_df.shape[1]} columns"
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            train_df.head(10),
            use_container_width=True
        )

        if "Churn" not in train_df.columns:

            st.error(
                "❌ This CSV does not contain a 'Churn' column."
            )

        else:

            if st.button(
                "🚀 Train XGBoost Model",
                use_container_width=True
            ):

                with st.spinner(
                    "Training model... Please wait."
                ):

                    model_package, metrics = train_model(
                        train_df
                    )

                if model_package is not None:

                    st.success(
                        "✅ Model trained and saved successfully!"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Accuracy",
                            f"{metrics['accuracy'] * 100:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Precision",
                            f"{metrics['precision'] * 100:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Recall",
                            f"{metrics['recall'] * 100:.2f}%"
                        )

                    with col4:
                        st.metric(
                            "F1 Score",
                            f"{metrics['f1'] * 100:.2f}%"
                        )

                    st.subheader("Confusion Matrix")

                    cm = metrics["confusion_matrix"]

                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual No", "Actual Yes"],
                        columns=["Predicted No", "Predicted Yes"]
                    )

                    st.dataframe(
                        cm_df,
                        use_container_width=True
                    )


# =========================================================
# CSV PREDICTION
# =========================================================

elif page == "📂 CSV Prediction":

    st.subheader("📂 Predict Churn from Another CSV")

    st.write(
        """
        Upload another customer CSV here.

        The CSV should contain the same type of customer information
        used during training. The **Churn column is optional**.
        """
    )

    model_package = load_saved_model()

    if model_package is None:

        st.warning(
            "⚠️ No trained model found. "
            "Go to **Train Model** and train the model first."
        )

    else:

        uploaded_prediction = st.file_uploader(
            "Upload Customer CSV for Prediction",
            type=["csv"],
            key="prediction_file"
        )

        if uploaded_prediction is not None:

            prediction_df = pd.read_csv(
                uploaded_prediction
            )

            st.success(
                f"CSV loaded: {prediction_df.shape[0]} customers"
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                prediction_df.head(10),
                use_container_width=True
            )

            if st.button(
                "🔮 Predict Churn",
                use_container_width=True
            ):

                model = model_package["model"]
                training_features = model_package["features"]

                try:

                    prepared_df = prepare_prediction_data(
                        prediction_df,
                        training_features
                    )

                    predictions = model.predict(
                        prepared_df
                    )

                    probabilities = model.predict_proba(
                        prepared_df
                    )[:, 1]

                    result_df = prediction_df.copy()

                    result_df["Prediction"] = np.where(
                        predictions == 1,
                        "Churn",
                        "Stay"
                    )

                    result_df["Churn_Probability"] = (
                        probabilities * 100
                    ).round(2)

                    st.success(
                        "✅ Prediction completed successfully!"
                    )

                    # Counts
                    churn_count = int(
                        (predictions == 1).sum()
                    )

                    stay_count = int(
                        (predictions == 0).sum()
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Customers",
                            len(result_df)
                        )

                    with col2:
                        st.metric(
                            "Predicted Churn",
                            churn_count
                        )

                    with col3:
                        st.metric(
                            "Predicted Stay",
                            stay_count
                        )

                    st.subheader(
                        "📊 Prediction Results"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # Download
                    csv_output = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction CSV",
                        data=csv_output,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        "❌ Prediction failed."
                    )

                    st.write(
                        "Error details:"
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# MANUAL PREDICTION
# =========================================================

elif page == "👤 Manual Prediction":

    st.subheader("👤 Manual Customer Prediction")

    model_package = load_saved_model()

    if model_package is None:

        st.warning(
            "⚠️ Please train the model first."
        )

    else:

        model = model_package["model"]
        training_features = model_package["features"]

        st.write(
            "Enter customer information below."
        )

        # -------------------------------------------------
        # BASIC INFORMATION
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col2:
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

        with col3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with col2:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

        with col3:
            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        # -------------------------------------------------
        # SERVICES
        # -------------------------------------------------

        st.subheader("Internet & Services")

        col1, col2, col3 = st.columns(3)

        with col1:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

        with col2:
            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

        with col3:
            online_security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            online_backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:
            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with col2:
            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        # -------------------------------------------------
        # BILLING
        # -------------------------------------------------

        st.subheader("Billing Information")

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

        with col2:
            paperless = st.selectbox(
                "Paperless Billing",
                [
                    "Yes",
                    "No"
                ]
            )

        with col3:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

        with col2:
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=1000.0
            )

        # -------------------------------------------------
        # CREATE CUSTOMER DATA
        # -------------------------------------------------

        customer = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }]
        )

        if st.button(
            "🔮 Predict Customer Churn",
            use_container_width=True
        ):

            try:

                customer = customer[training_features]

                prediction = model.predict(
                    customer
                )[0]

                probability = model.predict_proba(
                    customer
                )[0][1]

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction == 1:

                        st.error(
                            "⚠️ Customer is predicted to CHURN"
                        )

                    else:

                        st.success(
                            "✅ Customer is predicted to STAY"
                        )

                with col2:

                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )

                st.progress(
                    float(probability)
                )

            except Exception as e:

                st.error(
                    "❌ Manual prediction failed."
                )

                st.code(
                    str(e)
                )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "ℹ️ Model Information":

    st.subheader("ℹ️ Model Information")

    st.write(
        """
        ### Algorithm
        **XGBoost (Extreme Gradient Boosting)**

        XGBoost is a machine learning algorithm based on
        decision trees. It is used here as a binary classification
        model to predict whether a customer will churn or stay.

        ### Input
        Customer demographic, service, tenure and billing information.

        ### Output
        - Churn
        - Stay
        - Churn Probability

        ### Model Workflow

        CSV Dataset
        ↓
        Data Cleaning
        ↓
        Missing Value Handling
        ↓
        Categorical Encoding
        ↓
        Train-Test Split
        ↓
        XGBoost
        ↓
        Prediction
        ↓
        Churn / Stay
        """
    )

    model_package = load_saved_model()

    if model_package is not None:

        st.subheader("Training Features")

        features = model_package["features"]

        st.write(
            f"Number of features: **{len(features)}**"
        )

        st.dataframe(
            pd.DataFrame(
                {"Feature": features}
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Train the model to view the feature information."
        )
