import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LIGHT PROFESSIONAL THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f4f8fc;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d9e4ef;
    }

    [data-testid="stSidebar"] * {
        color: #173f70;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    h1 {
        color: #123b70 !important;
        font-weight: 850 !important;
    }

    h2 {
        color: #174f7c !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #176b76 !important;
        font-weight: 750 !important;
    }

    .stButton button {
        font-size: 17px !important;
        font-weight: 750 !important;
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


try:

    model = load_model()

except Exception as e:

    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()


# Get feature names directly from trained XGBoost model
model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Customer Churn AI")

st.sidebar.caption(
    "XGBoost Customer Analytics System"
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
    "Enter customer information to estimate "
    "the probability of service churn."
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("📊 Customer Churn AI")

    st.subheader(
        "XGBoost Based Customer Churn Analysis System"
    )

    st.write(
        "A machine learning system that estimates "
        "customer churn probability using customer "
        "profile, service and billing information."
    )

    st.divider()

    # Overview

    st.header("📌 System Overview")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "🤖 Machine Learning",
            "XGBoost"
        )

    with c2:

        st.metric(
            "🎯 Task",
            "Binary Classification"
        )

    with c3:

        st.metric(
            "📊 Prediction",
            "Churn / Stay"
        )

    st.divider()

    # How it works

    st.header("⚙️ How It Works")

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
            "The trained XGBoost model processes "
            "the customer information."
        )

    with h3:

        st.subheader("03 · 🎯 Prediction")

        st.write(
            "The system estimates churn probability "
            "and displays the prediction."
        )

    st.divider()

    st.info(
        "💡 Customer churn refers to customers "
        "discontinuing a service. This system uses "
        "machine learning to estimate churn probability."
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")

    st.write(
        "Enter the customer's information below "
        "to generate an XGBoost churn prediction."
    )

    st.divider()

    # =====================================================
    # CUSTOMER DETAILS
    # =====================================================

    st.header("👤 Customer Profile")

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

    with col2:

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col3:

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


    st.divider()


    # =====================================================
    # INTERNET & SERVICES
    # =====================================================

    st.header("🌐 Internet & Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col2:

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    col4, col5 = st.columns(2)

    with col4:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

    with col5:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


    st.divider()


    # =====================================================
    # CONTRACT & BILLING
    # =====================================================

    st.header("💳 Contract & Billing")

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


    st.divider()


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.header("🔮 Generate Prediction")

    predict_button = st.button(
        "🚀 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # PREDICTION
    # =====================================================

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

            st.divider()

            st.header("📊 Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ HIGH CHURN RISK"
                )

                st.subheader(
                    "⚠️ Customer is likely to CHURN"
                )

                st.write(
                    "The trained XGBoost model estimates "
                    "a higher probability of service churn."
                )

            else:

                st.success(
                    "✅ LOW CHURN RISK"
                )

                st.subheader(
                    "✅ Customer is likely to STAY"
                )

                st.write(
                    "The trained XGBoost model estimates "
                    "a lower probability of service churn."
                )


            # =================================================
            # GAUGE
            # =================================================

            st.subheader("🎯 Churn Probability")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percent,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 36
                        }
                    },

                    title={
                        "text": "Estimated Churn Probability"
                    },

                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "color": "#1769AA"
                        },

                        "steps": [
                            {
                                "range": [0, 40],
                                "color": "#DDF5E5"
                            },
                            {
                                "range": [40, 70],
                                "color": "#FFF1CC"
                            },
                            {
                                "range": [70, 100],
                                "color": "#FFE0E0"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=300,
                margin=dict(
                    l=30,
                    r=30,
                    t=60,
                    b=20
                ),
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                gauge,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )


            # =================================================
            # METRICS
            # =================================================

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


            # =================================================
            # PROGRESS
            # =================================================

            st.subheader("📊 Probability Level")

            st.progress(
                float(probability)
            )

            st.caption(
                f"Estimated churn probability: "
                f"{probability_percent:.2f}%"
            )


            # =================================================
            # CUSTOMER SUMMARY
            # =================================================

            with st.expander(
                "👁️ View Customer Input Summary"
            ):

                s1, s2, s3 = st.columns(3)

                with s1:

                    st.write(
                        f"**Gender:** {gender}"
                    )

                    st.write(
                        f"**Tenure:** {tenure} months"
                    )

                    st.write(
                        f"**Partner:** {partner}"
                    )

                    st.write(
                        f"**Dependents:** {dependents}"
                    )

                with s2:

                    st.write(
                        f"**Internet:** {internet}"
                    )

                    st.write(
                        f"**Online Security:** {security}"
                    )

                    st.write(
                        f"**Tech Support:** {support}"
                    )

                    st.write(
                        f"**Streaming TV:** {streaming_tv}"
                    )

                with s3:

                    st.write(
                        f"**Contract:** {contract}"
                    )

                    st.write(
                        f"**Payment:** {payment}"
                    )

                    st.write(
                        f"**Monthly Charges:** ${monthly:.2f}"
                    )

                    st.write(
                        f"**Total Charges:** ${total:.2f}"
                    )


        except Exception as e:

            st.error(
                "Prediction error occurred."
            )

            st.code(str(e))


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title("🤖 XGBoost Model")

    st.write(
        "Information about the trained customer churn "
        "classification model."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("🧠 Model Details")

        st.write(
            "**Algorithm:** XGBoost Classifier"
        )

        st.write(
            "**Task:** Binary Classification"
        )

        st.write(
            "**Output:** Churn / Stay"
        )

        st.write(
            "**Prediction:** Churn Probability"
        )

    with c2:

        st.subheader("📊 Input Categories")

        st.write(
            "• Customer profile"
        )

        st.write(
            "• Internet services"
        )

        st.write(
            "• Additional services"
        )

        st.write(
            "• Contract information"
        )

        st.write(
            "• Billing information"
        )


    st.divider()

    st.subheader("🔧 Prediction Process")

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

    st.info(
        f"The trained XGBoost model contains "
        f"{len(model_features)} processed input features."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn AI • XGBoost Classification System"
)
