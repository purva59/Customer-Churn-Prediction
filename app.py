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
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROFESSIONAL LIGHT BACKGROUND
# ============================================================

st.markdown("""
<style>

    /* ================================
       MAIN BACKGROUND
       ================================ */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 10%,
                rgba(99, 102, 241, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 15%,
                rgba(14, 165, 233, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(139, 92, 246, 0.10),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #f8fbff 0%,
                #eef5ff 45%,
                #f8f7ff 100%
            );
    }


    /* ================================
       PAGE WIDTH
       ================================ */

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ================================
       HEADER
       ================================ */

    h1 {
        color: #172554 !important;
        font-size: 46px !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        margin-bottom: 0 !important;
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


    /* ================================
       TOP HEADER AREA
       ================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(203, 213, 225, 0.75);
        border-radius: 20px;
        box-shadow:
            0 10px 35px rgba(30, 64, 175, 0.08);
    }


    /* ================================
       INPUT LABELS
       ================================ */

    label {
        color: #334155 !important;
        font-weight: 650 !important;
    }


    /* ================================
       SELECT BOX
       ================================ */

    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid #d7e0ed !important;
        border-radius: 11px !important;
        min-height: 44px;
    }


    /* ================================
       NUMBER INPUT
       ================================ */

    div[data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid #d7e0ed !important;
        border-radius: 11px !important;
        color: #172554 !important;
    }


    /* ================================
       BUTTON
       ================================ */

    div.stButton > button {
        min-height: 58px;
        border: none;
        border-radius: 14px;

        background:
            linear-gradient(
                100deg,
                #2563eb,
                #4f46e5,
                #7c3aed
            );

        color: white !important;

        font-size: 17px;
        font-weight: 800;

        box-shadow:
            0 10px 25px rgba(79, 70, 229, 0.25);

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 14px 30px rgba(79, 70, 229, 0.32);
    }


    /* ================================
       METRICS
       ================================ */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px;

        box-shadow:
            0 6px 20px rgba(15,23,42,0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 800 !important;
    }


    /* ================================
       ALERT BOXES
       ================================ */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }


    /* ================================
       DIVIDER
       ================================ */

    hr {
        border: none;
        border-top: 1px solid rgba(148,163,184,0.25);
        margin: 28px 0;
    }


    /* ================================
       EXPANDER
       ================================ */

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOUR EXISTING XGBOOST MODEL
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


# ============================================================
# GET FEATURES DIRECTLY FROM TRAINED MODEL
# ============================================================

model_features = model.get_booster().feature_names


if model_features is None:

    st.error(
        "The trained XGBoost model does not contain feature names."
    )

    st.stop()


# ============================================================
# HERO SECTION
# ============================================================

with st.container(border=True):

    st.caption("✦ ARTIFICIAL INTELLIGENCE  •  CUSTOMER ANALYTICS")

    st.title("Customer Churn AI")

    st.write(
        "Predict customer churn probability using your trained "
        "XGBoost classification model."
    )

    st.write("")


    hero1, hero2, hero3 = st.columns(3)

    with hero1:

        st.metric(
            "MODEL",
            "XGBoost"
        )

    with hero2:

        st.metric(
            "TASK",
            "Churn Prediction"
        )

    with hero3:

        st.metric(
            "ANALYSIS",
            "Customer Risk"
        )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.write("")

st.subheader("👤 Customer Profile")

with st.container(border=True):

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
            ["Yes", "No", "No phone service"]
        )


# ============================================================
# INTERNET & SERVICES
# ============================================================

st.write("")

st.subheader("🌐 Internet & Services")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

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

    with col2:

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

    with col3:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.write("")

st.subheader("💳 Contract & Billing")

with st.container(border=True):

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


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")
st.write("")

predict_button = st.button(
    "🔮  ANALYZE CUSTOMER CHURN",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # ORIGINAL CUSTOMER DATA
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SAME ENCODING USED IN YOUR ORIGINAL APP
    # --------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # --------------------------------------------------------
    # MATCH EXACT TRAINING FEATURES
    # --------------------------------------------------------

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )


    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    customer_encoded = customer_encoded.astype(float)


    try:

        # ----------------------------------------------------
        # SAME MODEL PREDICTION
        # ----------------------------------------------------

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

        st.write("")
        st.divider()

        st.subheader("🎯 AI Prediction")


        if prediction == 1:

            st.error("⚠️ HIGH CHURN RISK")

            result_text = "Customer is likely to CHURN"

            explanation = (
                "The XGBoost model has classified this customer "
                "as a higher churn-risk case."
            )

        else:

            st.success("✓ LOW CHURN RISK")

            result_text = "Customer is likely to STAY"

            explanation = (
                "The XGBoost model has classified this customer "
                "as a lower churn-risk case."
            )


        # ====================================================
        # MAIN RESULT CARD
        # ====================================================

        with st.container(border=True):

            result_col1, result_col2 = st.columns(
                [1.1, 1]
            )


            with result_col1:

                st.subheader(result_text)

                st.write(explanation)

                st.write("")

                st.metric(
                    "Estimated Churn Probability",
                    f"{probability_percent:.2f}%"
                )


            # =================================================
            # GAUGE
            # =================================================

            with result_col2:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",

                        value=probability_percent,

                        number={
                            "suffix": "%",
                            "font": {
                                "size": 38,
                                "color": "#172554"
                            }
                        },

                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1,
                                "tickcolor": "#94a3b8"
                            },

                            "bar": {
                                "color": "#4f46e5",
                                "thickness": 0.72
                            },

                            "bgcolor": "#edf2f7",

                            "borderwidth": 1,

                            "bordercolor": "#dbe3ef"
                        }
                    )
                )


                gauge.update_layout(
                    height=260,

                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=10
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


        # ====================================================
        # PREDICTION SUMMARY
        # ====================================================

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


        # ====================================================
        # CUSTOMER SNAPSHOT
        # ====================================================

        st.write("")

        st.subheader("👁️ Customer Snapshot")

        snap1, snap2, snap3, snap4 = st.columns(4)


        with snap1:

            st.info(
                f"**Internet Service**\n\n{internet}"
            )


        with snap2:

            st.info(
                f"**Monthly Charges**\n\n${monthly:.2f}"
            )


        with snap3:

            st.info(
                f"**Payment Method**\n\n{payment}"
            )


        with snap4:

            st.info(
                f"**Support**\n\n{support}"
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.write("")

        with st.expander(
            "ℹ️ How to interpret this result"
        ):

            st.write(
                "The XGBoost model uses the customer information "
                "provided above to classify the customer into "
                "the churn or stay category."
            )

            st.write(
                "The displayed percentage is the model's estimated "
                "probability for the churn class."
            )

            st.write(
                "This prediction is based on the trained model and "
                "should be interpreted as a machine-learning estimate, "
                "not as a guaranteed future outcome."
            )


    except Exception as e:

        st.error(
            "❌ Prediction error occurred."
        )

        st.code(str(e))
