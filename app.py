import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL LIGHT THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(135deg,
        #f7fbff 0%,
        #eef6ff 45%,
        #f9fcff 100%);
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #edf6ff 100%
    );
    border-right: 1px solid #d9e7f5;
}

[data-testid="stSidebar"] h1 {
    color: #164e78 !important;
    font-size: 25px !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #214c6f !important;
}

/* Main headings */

h1 {
    color: #123f63 !important;
    font-weight: 850 !important;
}

h2 {
    color: #155e75 !important;
    font-weight: 800 !important;
}

h3 {
    color: #245b78 !important;
    font-weight: 750 !important;
}

/* Input labels */

.stSelectbox label,
.stNumberInput label,
.stFileUploader label {
    color: #294b63 !important;
    font-weight: 650 !important;
}

/* Buttons */

.stButton > button {
    border-radius: 11px !important;
    font-size: 17px !important;
    font-weight: 750 !important;
    min-height: 48px !important;
}

/* Metrics */

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid #dce8f3;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(40,80,120,0.08);
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
}

/* Divider */

hr {
    border-color: #dce8f3 !important;
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

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Customer Churn AI")

st.sidebar.caption(
    "Customer Analytics & Prediction"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🎯 Churn Prediction",
        "🤖 Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Powered by XGBoost Machine Learning"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("📊 Customer Churn Prediction")

    st.subheader(
        "XGBoost Based Customer Churn Analysis System"
    )

    st.write(
        "Analyze customer information and estimate "
        "the probability of service churn using the "
        "trained XGBoost classification model."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "🤖 Algorithm",
            "XGBoost"
        )

    with c2:

        st.metric(
            "🎯 Task",
            "Classification"
        )

    with c3:

        st.metric(
            "📊 Output",
            "Churn / Stay"
        )

    st.divider()

    st.header("⚙️ How the System Works")

    h1, h2, h3 = st.columns(3)

    with h1:

        st.subheader("01 · 👤 Customer Data")

        st.write(
            "Enter customer profile, service, "
            "contract and billing information."
        )

    with h2:

        st.subheader("02 · 🧠 XGBoost")

        st.write(
            "The existing trained XGBoost model "
            "analyzes the customer information."
        )

    with h3:

        st.subheader("03 · 🎯 Prediction")

        st.write(
            "The system generates churn probability "
            "and predicts Churn or Stay."
        )

    st.divider()

    st.info(
        "💡 Use the Churn Prediction section for "
        "individual customer prediction or CSV batch prediction."
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")

    st.write(
        "Choose how you want to generate the prediction."
    )

    st.divider()

    manual_tab, csv_tab = st.tabs(
        [
            "👤 Manual Customer Prediction",
            "📂 CSV File Prediction"
        ]
    )


    # =====================================================
    # MANUAL PREDICTION
    # =====================================================

    with manual_tab:

        st.header("👤 Customer Profile")

        st.caption(
            "Enter customer details to generate an AI-powered churn prediction."
        )

        st.divider()


        # -------------------------------------------------
        # CUSTOMER DETAILS
        # -------------------------------------------------

        st.subheader("👤 Customer Details")

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
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )


        st.divider()


        # -------------------------------------------------
        # INTERNET SERVICES
        # -------------------------------------------------

        st.subheader("🌐 Internet & Services")

        internet_col1, internet_col2, internet_col3 = st.columns(3)

        with internet_col1:

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

        with internet_col2:

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with internet_col3:

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


        st.divider()


        # -------------------------------------------------
        # BILLING
        # -------------------------------------------------

        st.subheader("💳 Contract & Billing")

        bill1, bill2, bill3 = st.columns(3)

        with bill1:

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
                [
                    "Yes",
                    "No"
                ]
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


        st.divider()


        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        st.subheader("🔮 AI Prediction")

        predict_button = st.button(
            "🚀 Generate Customer Churn Prediction",
            type="primary",
            use_container_width=True
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

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


            # Same preprocessing
            customer_encoded = pd.get_dummies(
                customer,
                drop_first=True
            )


            customer_encoded = customer_encoded.reindex(
                columns=model_features,
                fill_value=0
            )


            customer_encoded = customer_encoded.astype(float)


            try:

                prediction = model.predict(
                    customer_encoded
                )[0]

                probability = model.predict_proba(
                    customer_encoded
                )[0][1]

                probability_percent = probability * 100


                # =================================================
                # ATTRACTIVE RESULT
                # =================================================

                st.divider()

                st.header("📌 Prediction Result")


                if prediction == 1:

                    st.error(
                        "⚠️ HIGH CHURN RISK"
                    )

                    st.subheader(
                        "⚠️ Customer is likely to CHURN"
                    )

                    st.write(
                        "The model predicts a higher probability "
                        "of customer churn."
                    )

                else:

                    st.success(
                        "✅ LOW CHURN RISK"
                    )

                    st.subheader(
                        "✅ Customer is likely to STAY"
                    )

                    st.write(
                        "The model predicts a higher probability "
                        "of customer retention."
                    )


                # -------------------------------------------------
                # MAIN METRICS
                # -------------------------------------------------

                st.write("### 📊 Prediction Analysis")

                r1, r2, r3 = st.columns(3)

                with r1:

                    st.metric(
                        "🎯 Churn Probability",
                        f"{probability_percent:.1f}%"
                    )

                with r2:

                    st.metric(
                        "🛡️ Stay Probability",
                        f"{100 - probability_percent:.1f}%"
                    )

                with r3:

                    risk_status = (
                        "Higher Risk"
                        if probability >= 0.5
                        else "Lower Risk"
                    )

                    st.metric(
                        "📍 Risk Status",
                        risk_status
                    )


                # -------------------------------------------------
                # GAUGE
                # -------------------------------------------------

                st.write("### 🎯 Churn Probability Analysis")

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability_percent,

                        number={
                            "suffix": "%",
                            "font": {
                                "size": 42,
                                "color": "#173f5f"
                            }
                        },

                        title={
                            "text": "Estimated Churn Probability"
                        },

                        gauge={

                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1
                            },

                            "bar": {
                                "color": "#1976a8",
                                "thickness": 0.25
                            },

                            "bgcolor": "#ffffff",

                            "borderwidth": 1,

                            "bordercolor": "#d7e3ed",

                            "steps": [

                                {
                                    "range": [0, 40],
                                    "color": "#dff5e8"
                                },

                                {
                                    "range": [40, 70],
                                    "color": "#fff2cc"
                                },

                                {
                                    "range": [70, 100],
                                    "color": "#ffe2e2"
                                }

                            ],

                            "threshold": {

                                "line": {
                                    "color": "#d9534f",
                                    "width": 4
                                },

                                "thickness": 0.8,

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
                    ),
                    paper_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )


                # -------------------------------------------------
                # INSIGHT
                # -------------------------------------------------

                st.write("### 🔍 Prediction Insight")

                st.info(
                    "The XGBoost model evaluates the customer's "
                    "service, contract, tenure and billing information "
                    "to generate a churn probability."
                )

                st.write(
                    f"**Churn probability: "
                    f"{probability_percent:.1f}%**"
                )


                # -------------------------------------------------
                # CUSTOMER SUMMARY
                # -------------------------------------------------

                with st.expander(
                    "📋 View Customer Information"
                ):

                    s1, s2, s3 = st.columns(3)

                    with s1:

                        st.write("### 👤 Customer")

                        st.write(
                            f"**Gender:** {gender}"
                        )

                        st.write(
                            f"**Senior Citizen:** {senior}"
                        )

                        st.write(
                            f"**Partner:** {partner}"
                        )

                        st.write(
                            f"**Dependents:** {dependents}"
                        )

                        st.write(
                            f"**Tenure:** {tenure} months"
                        )

                        st.write(
                            f"**Phone Service:** {phone}"
                        )

                    with s2:

                        st.write("### 🌐 Services")

                        st.write(
                            f"**Internet:** {internet}"
                        )

                        st.write(
                            f"**Online Security:** {security}"
                        )

                        st.write(
                            f"**Online Backup:** {backup}"
                        )

                        st.write(
                            f"**Device Protection:** {protection}"
                        )

                        st.write(
                            f"**Tech Support:** {support}"
                        )

                    with s3:

                        st.write("### 💳 Billing")

                        st.write(
                            f"**Contract:** {contract}"
                        )

                        st.write(
                            f"**Payment Method:** {payment}"
                        )

                        st.write(
                            f"**Monthly Charges:** ${monthly:.2f}"
                        )

                        st.write(
                            f"**Total Charges:** ${total:.2f}"
                        )

                        st.write(
                            f"**Paperless Billing:** {paperless}"
                        )


            except Exception as e:

                st.error(
                    "❌ Prediction error occurred."
                )

                st.code(str(e))


    # =====================================================
    # CSV PREDICTION
    # =====================================================

    with csv_tab:

        st.header("📂 CSV File Prediction")

        st.write(
            "Upload your customer CSV file to generate "
            "predictions for multiple customers."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="Upload a CSV containing customer information."
        )

        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    "✅ CSV uploaded successfully."
                )

                st.subheader("👀 Uploaded Data")

                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True
                )

                st.caption(
                    f"Total records: {len(uploaded_df)} | "
                    f"Columns: {len(uploaded_df.columns)}"
                )


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
                        "❌ Required columns are missing."
                    )

                    st.write(
                        missing_columns
                    )

                else:

                    st.divider()

                    predict_csv = st.button(
                        "🚀 Generate CSV Predictions",
                        type="primary",
                        use_container_width=True
                    )


                    if predict_csv:

                        result_df = uploaded_df.copy()

                        prediction_df = uploaded_df.copy()


                        if "customerID" in prediction_df.columns:

                            prediction_df = prediction_df.drop(
                                "customerID",
                                axis=1
                            )


                        if "Churn" in prediction_df.columns:

                            prediction_df = prediction_df.drop(
                                "Churn",
                                axis=1
                            )


                        prediction_df["TotalCharges"] = pd.to_numeric(
                            prediction_df["TotalCharges"],
                            errors="coerce"
                        )


                        prediction_df["TotalCharges"] = (
                            prediction_df["TotalCharges"]
                            .fillna(
                                prediction_df["TotalCharges"].median()
                            )
                        )


                        encoded_df = pd.get_dummies(
                            prediction_df,
                            drop_first=True
                        )


                        encoded_df = encoded_df.reindex(
                            columns=model_features,
                            fill_value=0
                        )


                        encoded_df = encoded_df.astype(float)


                        predictions = model.predict(
                            encoded_df
                        )


                        probabilities = model.predict_proba(
                            encoded_df
                        )[:, 1]


                        result_df["Predicted_Churn"] = [

                            "Churn"
                            if value == 1
                            else "Stay"

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


                        st.divider()

                        st.subheader(
                            "📊 Prediction Summary"
                        )


                        total_customers = len(
                            result_df
                        )


                        churn_count = (
                            result_df[
                                "Predicted_Churn"
                            ]
                            .eq("Churn")
                            .sum()
                        )


                        stay_count = (
                            result_df[
                                "Predicted_Churn"
                            ]
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
                                "📈 Avg. Probability",
                                f"{average_probability:.2f}%"
                            )


                        st.divider()


                        st.subheader(
                            "📋 Prediction Results"
                        )


                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )


                        csv_bytes = result_df.to_csv(
                            index=False
                        ).encode("utf-8")


                        st.download_button(
                            "⬇️ Download Prediction Results",
                            data=csv_bytes,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


            except Exception as e:

                st.error(
                    "❌ Error while processing CSV."
                )

                st.code(
                    str(e)
                )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.subheader(
        "XGBoost Customer Churn Classification Model"
    )

    st.write(
        "The application uses the existing trained XGBoost "
        "model to estimate customer churn."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "🤖 Algorithm",
            "XGBoost"
        )

    with c2:

        st.metric(
            "🎯 Task",
            "Classification"
        )

    with c3:

        st.metric(
            "📊 Output",
            "Churn / Stay"
        )

    with c4:

        st.metric(
            "📋 Features",
            len(model_features)
        )

    st.divider()

    st.subheader("⚙️ Prediction Process")

    st.write(
        "1. Customer information is collected."
    )

    st.write(
        "2. Categorical variables are encoded."
    )

    st.write(
        "3. Features are aligned with the trained model."
    )

    st.write(
        "4. XGBoost generates the prediction."
    )

    st.write(
        "5. Churn probability is displayed."
    )

    st.divider()

    with st.expander(
        "🔍 View Model Features"
    ):

        st.write(
            model_features
        )

    st.info(
        "The existing churn_model.json is used directly. "
        "The application does not retrain the model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn AI • XGBoost Machine Learning"
)
