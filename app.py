import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LIGHT PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 8% 8%,
        rgba(99,102,241,0.16), transparent 25%),
        radial-gradient(circle at 92% 12%,
        rgba(14,165,233,0.14), transparent 28%),
        radial-gradient(circle at 50% 95%,
        rgba(168,85,247,0.10), transparent 30%),
        linear-gradient(135deg,#f8fbff,#eef5ff,#faf8ff);
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1 {
    color: #172554 !important;
    font-size: 44px !important;
    font-weight: 800 !important;
}

h2 {
    color: #172554 !important;
    font-weight: 800 !important;
}

h3 {
    color: #1e3a8a !important;
    font-weight: 750 !important;
}

p {
    color: #475569;
}

label {
    color: #334155 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 10px !important;
    min-height: 43px;
}

div[data-testid="stNumberInput"] input {
    background: white !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 10px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(203,213,225,0.8);
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(30,64,175,0.07);
}

div.stButton > button {
    min-height: 56px;
    border: none;
    border-radius: 13px;
    background: linear-gradient(
        100deg,#2563eb,#4f46e5,#7c3aed
    );
    color: white !important;
    font-size: 17px;
    font-weight: 800;
    box-shadow: 0 10px 25px rgba(79,70,229,0.25);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(79,70,229,0.32);
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 17px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #172554 !important;
    font-weight: 800 !important;
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

hr {
    border: none;
    border-top: 1px solid rgba(148,163,184,0.25);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
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


model_features = model.get_booster().feature_names

if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📊 Customer Churn AI")

    st.caption("XGBoost Customer Risk Analysis")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🎯 Churn Prediction",
            "🤖 Model Information"
        ]
    )

    st.divider()

    st.info(
        "Enter customer information and use the trained "
        "XGBoost model to estimate churn probability."
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    with st.container(border=True):

        st.caption(
            "✦ ARTIFICIAL INTELLIGENCE  •  CUSTOMER ANALYTICS"
        )

        st.title("Customer Churn AI")

        st.write(
            "An intelligent customer analytics application "
            "that estimates the probability of service churn "
            "using an XGBoost classification model."
        )

        st.write("")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("MODEL", "XGBoost")

        with c2:
            st.metric("TASK", "Classification")

        with c3:
            st.metric("ANALYSIS", "Customer Churn")

    st.write("")

    st.subheader("✨ How the system works")

    a, b, c = st.columns(3)

    with a:

        with st.container(border=True):

            st.subheader("01")
            st.write("Customer Data")
            st.caption(
                "Enter demographic, service and billing information."
            )

    with b:

        with st.container(border=True):

            st.subheader("02")
            st.write("AI Analysis")
            st.caption(
                "The trained XGBoost model processes the customer data."
            )

    with c:

        with st.container(border=True):

            st.subheader("03")
            st.write("Risk Prediction")
            st.caption(
                "The system displays the predicted churn class "
                "and probability."
            )

    st.write("")

    st.subheader("🚀 Start prediction")

    st.write(
        "Use the **Churn Prediction** section from the sidebar "
        "to analyze a customer."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")

    st.caption(
        "Enter customer details to generate an XGBoost churn prediction."
    )

    st.divider()


    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.subheader("👤 Customer Profile")

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:

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

        with c2:

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

        with c3:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )


    # ========================================================
    # SERVICES
    # ========================================================

    st.write("")

    st.subheader("🌐 Internet & Services")

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
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

        with c2:

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

        with c3:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )


    # ========================================================
    # BILLING
    # ========================================================

    st.write("")

    st.subheader("💳 Contract & Billing")

    with st.container(border=True):

        c1, c2, c3 = st.columns(3)

        with c1:

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

        with c2:

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with c3:

            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

            total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=800.0,
                step=10.0
            )


    # ========================================================
    # BUTTON
    # ========================================================

    st.write("")
    st.write("")

    predict_button = st.button(
        "🔮  ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

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


        # ====================================================
        # SAME ORIGINAL MODEL PIPELINE
        # ====================================================

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
            # RESULT
            # =================================================

            st.write("")
            st.divider()

            st.subheader("🎯 AI Prediction Result")

            if prediction == 1:

                st.error("⚠️ HIGH CHURN RISK")

                result_text = "Customer is likely to CHURN"

                description = (
                    "The trained XGBoost model has classified "
                    "this customer as churn."
                )

            else:

                st.success("✅ LOW CHURN RISK")

                result_text = "Customer is likely to STAY"

                description = (
                    "The trained XGBoost model has classified "
                    "this customer as stay."
                )


            # =================================================
            # RESULT CARD
            # =================================================

            with st.container(border=True):

                left, right = st.columns([1, 1])

                with left:

                    st.subheader(result_text)

                    st.write(description)

                    st.write("")

                    st.metric(
                        "Churn Probability",
                        f"{probability_percent:.2f}%"
                    )


                with right:

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",

                            value=probability_percent,

                            number={
                                "suffix": "%",
                                "font": {
                                    "size": 40,
                                    "color": "#172554"
                                }
                            },

                            title={
                                "text": "Estimated Churn Risk"
                            },

                            gauge={
                                "axis": {
                                    "range": [0, 100]
                                },

                                "bar": {
                                    "color": "#4f46e5"
                                },

                                "bgcolor": "#edf2f7",

                                "borderwidth": 1,

                                "bordercolor": "#dbe3ef"
                            }
                        )
                    )

                    fig.update_layout(
                        height=270,
                        margin=dict(
                            l=20,
                            r=20,
                            t=45,
                            b=10
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


            # =================================================
            # SUMMARY
            # =================================================

            st.write("")

            st.subheader("📌 Prediction Summary")

            s1, s2, s3, s4 = st.columns(4)

            with s1:

                st.metric(
                    "Prediction",
                    "CHURN" if prediction == 1 else "STAY"
                )

            with s2:

                st.metric(
                    "Probability",
                    f"{probability_percent:.2f}%"
                )

            with s3:

                st.metric(
                    "Contract",
                    contract
                )

            with s4:

                st.metric(
                    "Tenure",
                    f"{tenure} months"
                )


            # =================================================
            # CUSTOMER SNAPSHOT
            # =================================================

            st.write("")

            st.subheader("👁️ Customer Snapshot")

            x1, x2, x3, x4 = st.columns(4)

            with x1:

                st.info(
                    f"**Internet Service**\n\n{internet}"
                )

            with x2:

                st.info(
                    f"**Monthly Charges**\n\n${monthly:.2f}"
                )

            with x3:

                st.info(
                    f"**Payment Method**\n\n{payment}"
                )

            with x4:

                st.info(
                    f"**Tech Support**\n\n{support}"
                )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.write("")

            with st.expander(
                "ℹ️ About this prediction"
            ):

                st.write(
                    "The application uses the trained XGBoost "
                    "classification model to predict whether "
                    "the customer belongs to the churn class."
                )

                st.write(
                    "The displayed percentage represents the "
                    "model's estimated probability for churn."
                )


        except Exception as e:

            st.error("❌ Prediction error occurred.")

            st.code(str(e))


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.title("🤖 XGBoost Model")

    st.caption(
        "Information about the machine learning model used "
        "for customer churn prediction."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.subheader("🧠 Algorithm")

            st.write("XGBoost Classifier")

            st.write(
                "An advanced gradient boosting classification "
                "algorithm used by this application to estimate "
                "customer churn."
            )

    with c2:

        with st.container(border=True):

            st.subheader("🎯 Prediction")

            st.write("Binary Classification")

            st.write(
                "The model predicts one of two outcomes: "
                "customer churn or customer stay."
            )


    st.write("")

    with st.container(border=True):

        st.subheader("📋 Model Input Features")

        st.write(
            "The application uses the feature structure stored "
            "inside the trained XGBoost model."
        )

        st.write(
            f"Number of encoded model features: "
            f"**{len(model_features)}**"
        )

    st.write("")

    st.info(
        "The model itself is loaded from churn_model.json. "
        "This application does not retrain or modify the model."
    )
