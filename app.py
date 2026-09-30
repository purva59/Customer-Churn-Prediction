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
# PROFESSIONAL LIGHT UI
# =========================================================

st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.10), transparent 25%),
        radial-gradient(circle at 90% 15%, rgba(20,184,166,0.10), transparent 25%),
        linear-gradient(135deg, #f7faff 0%, #eef5ff 50%, #f8fbff 100%);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #ffffff 0%, #f1f7ff 100%);
    border-right: 1px solid #dbe7f5;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}


/* Sidebar title */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #123b70 !important;
    font-weight: 800 !important;
}


/* Sidebar navigation buttons */

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 12px;
    border: 1px solid #d8e5f5;
    background: #ffffff;
    color: #173f70;
    font-size: 17px;
    font-weight: 700;
    text-align: left;
    padding-left: 18px;
    margin-bottom: 8px;
    box-shadow: 0 3px 10px rgba(30, 80, 130, 0.06);
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #eaf3ff;
    border-color: #4f8edc;
    color: #0d4f9c;
    transform: translateX(3px);
}


/* ---------- HERO ---------- */

.hero-box {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f2f8ff 55%,
        #eafcff 100%
    );

    border: 1px solid #d8e8f7;
    border-radius: 24px;
    padding: 32px 38px;
    margin-bottom: 28px;

    box-shadow:
        0 12px 35px rgba(39, 92, 140, 0.10);
}

.hero-label {
    color: #147d78;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    color: #123b70;
    font-size: 42px;
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: 10px;
}

.hero-text {
    color: #536579;
    font-size: 17px;
    line-height: 1.6;
}


/* ---------- SECTION HEADINGS ---------- */

h1, h2, h3 {
    color: #173f70 !important;
}

h2 {
    font-weight: 850 !important;
}

h3 {
    font-weight: 750 !important;
}


/* ---------- CARDS ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.96);
    border: 1px solid #dce8f4;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(30, 80, 130, 0.07);
}


/* ---------- INPUT LABELS ---------- */

label {
    color: #274968 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}


/* ---------- INPUT BOXES ---------- */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid #cbdbea !important;
    background-color: #ffffff !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #4f8edc !important;
}


/* Number input */

div[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1px solid #cbdbea !important;
    background: #ffffff !important;
}


/* ---------- PRIMARY BUTTON ---------- */

.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #1769aa,
        #168b86
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;

    min-height: 52px;

    font-size: 17px !important;
    font-weight: 800 !important;

    box-shadow:
        0 7px 18px rgba(23, 105, 170, 0.22);

    transition: all 0.2s ease;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow:
        0 10px 24px rgba(23, 105, 170, 0.30);
}


/* ---------- NORMAL BUTTON ---------- */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}


/* ---------- METRICS ---------- */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce8f4;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(30,80,130,0.06);
}

div[data-testid="stMetricLabel"] {
    color: #62758a !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
    color: #123b70 !important;
    font-weight: 850 !important;
}


/* ---------- INFO / WARNING / SUCCESS ---------- */

div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 600;
}


/* ---------- DIVIDER ---------- */

hr {
    border: none;
    border-top: 1px solid #dbe7f3;
    margin: 25px 0;
}


/* ---------- PROGRESS ---------- */

div[data-testid="stProgressBar"] {
    height: 14px;
}


/* ---------- CAPTION ---------- */

.stCaption {
    color: #687b8f !important;
}


/* ---------- FOOTER ---------- */

.footer-text {
    text-align: center;
    color: #718399;
    font-size: 13px;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid #dbe7f3;
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


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("## 📊 Customer Churn AI")
st.sidebar.caption("XGBoost Prediction System")

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
    "Enter customer information to estimate "
    "the probability of service churn."
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero-box">

        <div class="hero-label">
            Artificial Intelligence • Customer Analytics
        </div>

        <div class="hero-title">
            📊 Customer Churn AI
        </div>

        <div class="hero-text">
            An XGBoost-based machine learning system designed
            to estimate customer churn probability using
            customer profile, service and billing information.
        </div>

    </div>
    """, unsafe_allow_html=True)


    # -------- OVERVIEW METRICS --------

    st.subheader("📌 System Overview")

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


    st.markdown("")


    # -------- HOW IT WORKS --------

    st.subheader("⚙️ How the System Works")

    h1, h2, h3 = st.columns(3)

    with h1:
        with st.container(border=True):
            st.markdown("### 01 · 👤 Customer Data")
            st.write(
                "Enter customer profile, service, contract "
                "and billing information."
            )

    with h2:
        with st.container(border=True):
            st.markdown("### 02 · 🧠 XGBoost Model")
            st.write(
                "The trained XGBoost model processes the "
                "customer information."
            )

    with h3:
        with st.container(border=True):
            st.markdown("### 03 · 🎯 Risk Prediction")
            st.write(
                "The system estimates churn probability "
                "and displays the prediction."
            )


    st.markdown("")

    with st.container(border=True):

        st.subheader("💡 About Customer Churn")

        st.write(
            "Customer churn refers to customers discontinuing "
            "a service. Predictive analytics can help identify "
            "customers who may be at higher risk of churn."
        )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🎯 Churn Prediction":

    st.markdown("""
    <div class="hero-box">

        <div class="hero-label">
            Prediction Workspace
        </div>

        <div class="hero-title">
            🎯 Customer Churn Prediction
        </div>

        <div class="hero-text">
            Enter the customer's information below and use
            the trained XGBoost model to estimate churn risk.
        </div>

    </div>
    """, unsafe_allow_html=True)


    # =====================================================
    # CUSTOMER DETAILS
    # =====================================================

    with st.container(border=True):

        st.subheader("👤 Customer Profile")
        st.caption(
            "Basic demographic and customer relationship information"
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


    # =====================================================
    # INTERNET & SERVICES
    # =====================================================

    with st.container(border=True):

        st.subheader("🌐 Internet & Services")
        st.caption(
            "Customer's subscribed internet and additional services"
        )

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


    # =====================================================
    # CONTRACT & BILLING
    # =====================================================

    with st.container(border=True):

        st.subheader("💳 Contract & Billing")
        st.caption(
            "Contract type, payment information and charges"
        )

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


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    st.markdown("")

    predict_button = st.button(
        "🚀  Predict Customer Churn",
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


            st.markdown("---")

            # =================================================
            # RESULT
            # =================================================

            st.subheader("📊 Prediction Result")

            result1, result2 = st.columns([1.2, 1])


            # -------- RESULT MESSAGE --------

            with result1:

                if prediction == 1:

                    st.error(
                        "⚠️ HIGH CHURN RISK"
                    )

                    st.markdown(
                        "### ⚠️ Customer is likely to **CHURN**"
                    )

                    st.write(
                        "The trained model estimates a higher "
                        "probability of service discontinuation."
                    )

                    st.warning(
                        "Consider reviewing this customer's "
                        "service and retention requirements."
                    )

                else:

                    st.success(
                        "✅ LOW CHURN RISK"
                    )

                    st.markdown(
                        "### ✅ Customer is likely to **STAY**"
                    )

                    st.write(
                        "The trained model estimates a lower "
                        "probability of service discontinuation."
                    )

                    st.info(
                        "The customer is predicted to continue "
                        "the service."
                    )


            # -------- GAUGE --------

            with result2:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability_percent,
                        number={
                            "suffix": "%",
                            "font": {
                                "size": 32,
                                "color": "#173f70"
                            }
                        },
                        title={
                            "text": "Churn Probability",
                            "font": {
                                "size": 18,
                                "color": "#536579"
                            }
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickwidth": 1
                            },
                            "bar": {
                                "color": "#1769aa"
                            },
                            "bgcolor": "#eef4fa",
                            "borderwidth": 1,
                            "bordercolor": "#d3e0ec",
                            "steps": [
                                {
                                    "range": [0, 40],
                                    "color": "#e8f7ef"
                                },
                                {
                                    "range": [40, 70],
                                    "color": "#fff6df"
                                },
                                {
                                    "range": [70, 100],
                                    "color": "#fdeaea"
                                }
                            ]
                        }
                    )
                )

                gauge.update_layout(
                    height=280,
                    margin=dict(
                        l=20,
                        r=20,
                        t=50,
                        b=20
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family="Arial"
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )


            # =================================================
            # SUMMARY
            # =================================================

            st.markdown("")

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
            # PROBABILITY
            # =================================================

            st.markdown("")

            with st.container(border=True):

                st.subheader("🎯 Churn Probability")

                st.progress(
                    float(probability)
                )

                st.caption(
                    f"Estimated churn probability: "
                    f"{probability_percent:.2f}%"
                )


            # =================================================
            # CUSTOMER SNAPSHOT
            # =================================================

            st.markdown("")

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

    st.markdown("""
    <div class="hero-box">

        <div class="hero-label">
            Machine Learning Model
        </div>

        <div class="hero-title">
            🤖 XGBoost Model
        </div>

        <div class="hero-text">
            Customer churn classification using the trained
            XGBoost model.
        </div>

    </div>
    """, unsafe_allow_html=True)


    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.subheader("🧠 Model")

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
                "**Probability:** Churn probability"
            )


    with c2:

        with st.container(border=True):

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


    st.markdown("")

    with st.container(border=True):

        st.subheader("🔧 Prediction Process")

        st.write(
            "1. Customer information is collected."
        )

        st.write(
            "2. Categorical information is encoded."
        )

        st.write(
            "3. Input features are aligned with the trained model."
        )

        st.write(
            "4. The XGBoost model generates the prediction."
        )

        st.write(
            "5. Churn probability is displayed."
        )


    st.markdown("")

    with st.container(border=True):

        st.subheader("📌 Model Features")

        st.write(
            f"The trained model uses "
            f"**{len(model_features)} input features** "
            f"after preprocessing."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-text">
        Customer Churn AI • XGBoost Classification System
        <br>
        Built for Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
