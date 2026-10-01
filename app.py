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
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL LIGHT UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(
            135deg,
            #f7fbff 0%,
            #eef6ff 45%,
            #f9fcff 100%
        );
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ffffff 0%,
        #f1f7fc 100%
    );
    border-right: 1px solid #dbe7f0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #173f5f;
}

h1 {
    color: #173f5f;
    font-weight: 800;
}

h2 {
    color: #205375;
    font-weight: 700;
}

h3 {
    color: #286b8f;
    font-weight: 650;
}

p, label {
    color: #34495e;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce8f1;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(30, 70, 100, 0.06);
}

div.stButton > button {
    background: linear-gradient(
        90deg,
        #1976a8,
        #286b8f
    );
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #155d82,
        #205375
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.card {
    background: white;
    border: 1px solid #dce8f1;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(30, 70, 100, 0.06);
}

.info-box {
    background: #eef7ff;
    border-left: 5px solid #1976a8;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

.success-box {
    background: #eefaf3;
    border-left: 5px solid #2e8b57;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

.warning-box {
    background: #fff8e6;
    border-left: 5px solid #e6a700;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

.danger-box {
    background: #fff0f0;
    border-left: 5px solid #d9534f;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}

.footer {
    text-align: center;
    color: #718096;
    padding: 30px 0 10px 0;
    font-size: 14px;
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

model_features = model.get_booster().feature_names


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 📊 Customer Churn")
st.sidebar.markdown("### Prediction System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🎯 Churn Prediction",
        "🤖 Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts whether a customer is likely to churn "
    "using an XGBoost classification model."
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("📊 Customer Churn Prediction")
    st.subheader("AI-Powered Customer Retention Analysis")

    st.markdown("""
    <div class="info-box">
    <b>Customer Churn Prediction System</b><br>
    This application uses customer information and an XGBoost machine
    learning model to estimate the probability of customer churn.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "🎯 Task",
            "Churn Classification"
        )

    with col3:
        st.metric(
            "📈 Output",
            "Churn Probability"
        )

    st.markdown("---")

    st.subheader("🔄 How the System Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
        <div class="card">
        <h3>1️⃣ Input</h3>
        Customer details are entered manually or uploaded using CSV.
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="card">
        <h3>2️⃣ Processing</h3>
        Data is encoded and arranged according to model features.
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="card">
        <h3>3️⃣ Prediction</h3>
        XGBoost predicts the customer's churn status.
        </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
        <div class="card">
        <h3>4️⃣ Result</h3>
        Churn probability and risk status are displayed.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📌 Key Features")

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        - 👤 Manual customer prediction
        - 📂 CSV batch prediction
        - 🎯 Churn probability
        - 📊 Risk status
        """)

    with f2:
        st.markdown("""
        - 📈 Visual prediction gauge
        - 🤖 XGBoost model
        - 🔍 Customer information
        - 📋 Prediction results table
        """)


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")

    st.markdown("""
    <div class="info-box">
    Select a prediction method below. You can either enter customer
    information manually or upload a CSV file for batch prediction.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(
        [
            "👤 Manual Customer Prediction",
            "📂 CSV File Prediction"
        ]
    )


    # =====================================================
    # MANUAL CUSTOMER PREDICTION
    # =====================================================

    with tab1:

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

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with col2:

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
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

        with col3:

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

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

        st.markdown("---")

        st.subheader("🌐 Internet & Services")

        col1, col2, col3 = st.columns(3)

        with col1:

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

        with col2:

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

        with col3:

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

        st.markdown("---")

        st.subheader("💳 Contract & Billing")

        col1, col2, col3 = st.columns(3)

        with col1:

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col2:

            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

        with col3:

            total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=800.0,
                step=10.0
            )

        st.markdown("---")

        st.subheader("🔮 AI Prediction")

        predict_button = st.button(
            "🚀 Predict Customer Churn",
            use_container_width=True
        )

        if predict_button:

            # =================================================
            # ORIGINAL PREDICTION LOGIC
            # =================================================

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

            customer_encoded = pd.get_dummies(
                customer,
                drop_first=True
            )

            customer_encoded = customer_encoded.reindex(
                columns=model_features,
                fill_value=0
            )

            customer_encoded = customer_encoded.astype(float)

            prediction = model.predict(
                customer_encoded
            )[0]

            probability = model.predict_proba(
                customer_encoded
            )[0][1]

            probability_percent = probability * 100


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.subheader("📌 Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ HIGH CHURN RISK"
                )

                st.subheader(
                    "⚠️ Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ LOW CHURN RISK"
                )

                st.subheader(
                    "✅ Customer is likely to STAY"
                )


            # =================================================
            # METRICS
            # =================================================

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


            # =================================================
            # GAUGE
            # =================================================

            st.markdown("---")

            gauge_col1, gauge_col2 = st.columns(
                [1, 1]
            )

            with gauge_col1:

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
                    height=400,
                    margin=dict(
                        l=20,
                        r=20,
                        t=70,
                        b=20
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            with gauge_col2:

                st.subheader("💡 Prediction Insight")

                if probability >= 0.5:

                    st.markdown("""
                    <div class="danger-box">
                    <b>Higher churn probability detected.</b><br><br>
                    The customer may require additional attention,
                    engagement or retention strategies.
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="success-box">
                    <b>Lower churn probability detected.</b><br><br>
                    The customer currently shows a lower estimated
                    likelihood of leaving the service.
                    </div>
                    """, unsafe_allow_html=True)


            # =================================================
            # CUSTOMER INFORMATION
            # =================================================

            st.markdown("---")

            with st.expander("📋 View Customer Information"):

                customer_display = pd.DataFrame({
                    "Feature": [
                        "Gender",
                        "Senior Citizen",
                        "Partner",
                        "Dependents",
                        "Tenure",
                        "Phone Service",
                        "Internet Service",
                        "Contract",
                        "Monthly Charges",
                        "Total Charges"
                    ],

                    "Value": [
                        gender,
                        senior,
                        partner,
                        dependents,
                        tenure,
                        phone,
                        internet,
                        contract,
                        monthly,
                        total
                    ]
                })

                st.dataframe(
                    customer_display,
                    use_container_width=True,
                    hide_index=True
                )


    # =====================================================
    # CSV FILE PREDICTION
    # =====================================================

    with tab2:

        st.subheader("📂 CSV File Prediction")

        st.markdown("""
        <div class="info-box">
        Upload a customer CSV file. The system will process all customers
        and display their churn predictions and probabilities.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📁 Upload Customer CSV File",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                df = pd.read_csv(uploaded_file)

                st.success(
                    f"✅ File uploaded successfully: {uploaded_file.name}"
                )

                st.markdown("---")

                st.subheader("👀 Uploaded Data Preview")

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                # =============================================
                # REQUIRED COLUMNS
                # =============================================

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

                # =============================================
                # REMOVE EXTRA COLUMNS
                # =============================================

                prediction_data = df.copy()

                if "customerID" in prediction_data.columns:
                    prediction_data = prediction_data.drop(
                        columns=["customerID"]
                    )

                if "Churn" in prediction_data.columns:
                    prediction_data = prediction_data.drop(
                        columns=["Churn"]
                    )

                # =============================================
                # CHECK REQUIRED COLUMNS
                # =============================================

                missing_columns = [
                    col
                    for col in required_columns
                    if col not in prediction_data.columns
                ]

                if missing_columns:

                    st.error(
                        "❌ Missing required columns:"
                    )

                    st.write(
                        missing_columns
                    )

                else:

                    # =========================================
                    # PREPARE DATA
                    # =========================================

                    prediction_data = prediction_data[
                        required_columns
                    ].copy()

                    prediction_data["TotalCharges"] = pd.to_numeric(
                        prediction_data["TotalCharges"],
                        errors="coerce"
                    )

                    prediction_data["TotalCharges"] = (
                        prediction_data["TotalCharges"]
                        .fillna(
                            prediction_data["TotalCharges"].median()
                        )
                    )

                    # =========================================
                    # ENCODING
                    # =========================================

                    encoded_data = pd.get_dummies(
                        prediction_data,
                        drop_first=True
                    )

                    encoded_data = encoded_data.reindex(
                        columns=model_features,
                        fill_value=0
                    )

                    encoded_data = encoded_data.astype(float)

                    # =========================================
                    # PREDICTION
                    # =========================================

                    predictions = model.predict(
                        encoded_data
                    )

                    probabilities = model.predict_proba(
                        encoded_data
                    )[:, 1]

                    # =========================================
                    # RESULT DATAFRAME
                    # =========================================

                    result_df = df.copy()

                    result_df["Predicted_Churn"] = predictions

                    result_df[
                        "Churn_Probability_Percent"
                    ] = probabilities * 100

                    result_df[
                        "Predicted_Churn"
                    ] = result_df[
                        "Predicted_Churn"
                    ].map(
                        {
                            0: "No",
                            1: "Yes"
                        }
                    )

                    # =========================================
                    # SUMMARY
                    # =========================================

                    st.markdown("---")

                    st.subheader("📊 Prediction Summary")

                    total_customers = len(result_df)

                    churn_customers = sum(
                        predictions == 1
                    )

                    stay_customers = sum(
                        predictions == 0
                    )

                    average_probability = (
                        probabilities.mean() * 100
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.metric(
                            "👥 Total Customers",
                            total_customers
                        )

                    with c2:

                        st.metric(
                            "⚠️ Predicted Churn",
                            churn_customers
                        )

                    with c3:

                        st.metric(
                            "✅ Predicted Stay",
                            stay_customers
                        )

                    with c4:

                        st.metric(
                            "🎯 Avg. Churn Probability",
                            f"{average_probability:.1f}%"
                        )

                    # =========================================
                    # FINAL RESULTS
                    # =========================================

                    st.markdown("---")

                    st.subheader(
                        "🔮 Customer Churn Predictions"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.success(
                        "✅ Prediction completed successfully."
                    )

                    st.info(
                        "📌 The prediction results are displayed above. "
                        "No download option is provided."
                    )


            except Exception as e:

                st.error(
                    f"❌ Error while processing CSV: {e}"
                )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.markdown("""
    <div class="info-box">
    This application uses an XGBoost classification model for
    customer churn prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🤖 Algorithm",
            "XGBoost"
        )

    with col2:

        st.metric(
            "🎯 Problem Type",
            "Classification"
        )

    with col3:

        st.metric(
            "📊 Features",
            len(model_features)
        )

    st.markdown("---")

    st.subheader("🔄 Prediction Process")

    st.markdown("""
    1. 👤 Customer information is collected.
    2. 🔤 Categorical values are converted using one-hot encoding.
    3. 🧩 Features are aligned with the trained model.
    4. 🤖 XGBoost performs the prediction.
    5. 📊 Churn probability is calculated.
    6. 🎯 Final churn status is displayed.
    """)

    st.markdown("---")

    st.subheader("📋 Model Features")

    feature_df = pd.DataFrame(
        {
            "Feature": model_features
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📌 Output")

    st.markdown("""
    The model provides:

    - **Predicted Churn Status**
    - **Churn Probability**
    - **Stay Probability**
    - **Risk Status**
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
Customer Churn Prediction System • XGBoost • Streamlit
</div>
""", unsafe_allow_html=True)
