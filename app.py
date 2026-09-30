import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        linear-gradient(135deg, #f8fbff 0%, #f2f6fc 50%, #f8f7ff 100%);
}

/* Main width */
.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}


/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e7edf5;
}

section[data-testid="stSidebar"] * {
    color: #344a63;
}


/* ---------------- HEADINGS ---------------- */

h1 {
    color: #102a43 !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px;
}

h2 {
    color: #183b56 !important;
    font-weight: 750 !important;
}

h3 {
    color: #234e70 !important;
    font-weight: 700 !important;
}


/* ---------------- TEXT ---------------- */

p {
    color: #62748a;
}


/* ---------------- INPUT LABEL ---------------- */

label {
    color: #334e68 !important;
    font-weight: 600 !important;
}


/* ---------------- SELECTBOX ---------------- */

div[data-baseweb="select"] > div {
    background: #ffffff;
    border: 1px solid #d8e2ee;
    border-radius: 10px;
}


/* ---------------- NUMBER INPUT ---------------- */

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #ffffff;
    border-radius: 10px;
}


/* ---------------- BUTTON ---------------- */

.stButton > button {

    background: linear-gradient(
        100deg,
        #1769e0,
        #6048d8
    );

    color: white;

    border: none;
    border-radius: 12px;

    min-height: 55px;

    font-size: 16px;
    font-weight: 800;

    box-shadow:
        0 10px 25px rgba(45, 87, 180, 0.22);

    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 15px 30px rgba(45, 87, 180, 0.30);
}


/* ---------------- METRIC CARDS ---------------- */

div[data-testid="stMetric"] {

    background: rgba(255,255,255,0.95);

    border: 1px solid #e2eaf3;

    border-radius: 16px;

    padding: 18px;

    box-shadow:
        0 6px 20px rgba(34, 68, 105, 0.06);
}


/* ---------------- ALERTS ---------------- */

div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* ---------------- EXPANDER ---------------- */

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 14px;
}


/* ---------------- DIVIDER ---------------- */

hr {
    border-color: #e2e8f0;
}


/* ---------------- DATAFRAME ---------------- */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}


/* ---------------- TABS ---------------- */

button[data-baseweb="tab"] {
    font-weight: 700;
}


/* ---------------- PROGRESS ---------------- */

div[data-testid="stProgress"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOUR EXISTING MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model("churn_model.json")

    return model


model = load_model()

model_features = model.get_booster().feature_names


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ◈ ChurnIQ")

    st.caption("Customer Intelligence System")

    st.divider()

    st.markdown("### MODEL")

    st.write("**Algorithm**")
    st.write("XGBoost Classifier")

    st.write("**Problem**")
    st.write("Binary Classification")

    st.write("**Prediction**")
    st.write("Customer Churn")

    st.divider()

    st.markdown("### ANALYTICS")

    st.write("01  Customer Profile")
    st.write("02  Service Usage")
    st.write("03  Billing Details")
    st.write("04  Churn Prediction")
    st.write("05  Risk Analysis")

    st.divider()

    st.success("● Model Ready")

    st.caption(
        "AI-powered customer retention analysis"
    )


# ============================================================
# HERO SECTION
# ============================================================

hero_left, hero_right = st.columns(
    [2.8, 1]
)

with hero_left:

    st.caption(
        "ARTIFICIAL INTELLIGENCE  •  CUSTOMER ANALYTICS"
    )

    st.title(
        "Customer Churn Intelligence"
    )

    st.write(
        "Analyze customer behavior and estimate the likelihood "
        "of service churn using your trained XGBoost classification model."
    )


with hero_right:

    st.metric(
        "MODEL",
        "XGBoost",
        "Classification"
    )


st.divider()


# ============================================================
# TOP STATUS CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "AI Engine",
        "XGBoost"
    )

with c2:
    st.metric(
        "Analysis",
        "Churn Risk"
    )

with c3:
    st.metric(
        "Output",
        "Probability"
    )

with c4:
    st.metric(
        "Mode",
        "Real-time"
    )


st.write("")


# ============================================================
# CUSTOMER INPUT AREA
# ============================================================

st.header("Customer Analysis")

st.caption(
    "Provide the customer's information to generate an AI-based churn assessment."
)


# ============================================================
# PROFILE
# ============================================================

with st.expander(
    "👤  CUSTOMER PROFILE",
    expanded=True
):

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with p2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with p3:

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

    with p4:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )


# ============================================================
# CUSTOMER VALUE
# ============================================================

with st.expander(
    "📊  CUSTOMER VALUE",
    expanded=True
):

    v1, v2, v3 = st.columns(3)

    with v1:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with v2:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

    with v3:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )


# ============================================================
# SERVICES
# ============================================================

with st.expander(
    "🌐  SERVICE USAGE",
    expanded=True
):

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

    with s2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

    with s3:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with s4:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    s1, s2, s3, s4 = st.columns(4)

    with s1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s4:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    s1, s2 = st.columns(2)

    with s1:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with s2:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


# ============================================================
# BILLING
# ============================================================

with st.expander(
    "💳  BILLING & PAYMENT",
    expanded=True
):

    b1, b2 = st.columns(2)

    with b1:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

    with b2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    analyze = st.button(
        "✦  RUN CHURN ANALYSIS",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    # --------------------------------------------------------
    # CUSTOMER DATA
    # --------------------------------------------------------

    customer = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

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


    # --------------------------------------------------------
    # SAME ENCODING LOGIC
    # --------------------------------------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)


    # --------------------------------------------------------
    # SAME MODEL
    # --------------------------------------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    stay_probability = 1 - probability


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.caption(
        "AI ANALYSIS COMPLETE"
    )

    st.header(
        "Prediction Overview"
    )


    # ========================================================
    # BIG RESULT
    # ========================================================

    result_col, gauge_col = st.columns(
        [1.15, 1]
    )


    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    with result_col:

        if prediction == 1:

            st.error(
                "HIGHER CHURN RISK"
            )

            st.subheader(
                "Customer is predicted to be at higher risk of churn."
            )

        else:

            st.success(
                "LOWER CHURN RISK"
            )

            st.subheader(
                "Customer is predicted to be at lower risk of churn."
            )

        st.write(
            "The prediction is generated directly from "
            "your trained XGBoost classification model."
        )

        st.write("")

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.1f}%"
            )

        with r2:

            st.metric(
                "Stay Probability",
                f"{stay_probability * 100:.1f}%"
            )


    # --------------------------------------------------------
    # GAUGE
    # --------------------------------------------------------

    with gauge_col:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=probability * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 38
                    }
                },

                title={
                    "text": "CHURN RISK SCORE",
                    "font": {
                        "size": 16
                    }
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#5b55d9"
                    },

                    "bgcolor": "#edf2f7",

                    "borderwidth": 0,

                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#dff4e8"
                        },

                        {
                            "range": [30, 60],
                            "color": "#fff2cc"
                        },

                        {
                            "range": [60, 100],
                            "color": "#ffe2e2"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=25,
                r=25,
                t=55,
                b=15
            ),

            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # PROBABILITY ANALYSIS
    # ========================================================

    st.subheader(
        "Probability Analysis"
    )

    chart_col, detail_col = st.columns(
        [1.5, 1]
    )


    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    with chart_col:

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=[
                    "Stay",
                    "Churn"
                ],

                y=[
                    stay_probability * 100,
                    probability * 100
                ],

                text=[
                    f"{stay_probability * 100:.1f}%",
                    f"{probability * 100:.1f}%"
                ],

                textposition="outside",

                marker=dict(
                    color=[
                        "#3b82f6",
                        "#7c5ce0"
                    ],

                    line=dict(
                        width=0
                    )
                )
            )
        )

        fig2.update_layout(

            height=370,

            yaxis=dict(
                title="Probability (%)",
                range=[0, 110]
            ),

            xaxis=dict(
                title=""
            ),

            template="plotly_white",

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            margin=dict(
                l=45,
                r=25,
                t=25,
                b=45
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------------
    # AI SUMMARY
    # --------------------------------------------------------

    with detail_col:

        st.subheader(
            "AI Summary"
        )

        st.info(
            "The XGBoost model has evaluated the customer's "
            "profile, service usage, contract and billing "
            "information."
        )

        st.write("")

        st.write(
            "**Prediction:** "
            + (
                "Higher churn risk"
                if prediction == 1
                else "Lower churn risk"
            )
        )

        st.write(
            f"**Churn probability:** "
            f"{probability * 100:.2f}%"
        )

        st.write(
            f"**Stay probability:** "
            f"{stay_probability * 100:.2f}%"
        )

        st.write(
            f"**Contract:** {contract}"
        )

        st.write(
            f"**Tenure:** {tenure} months"
        )

        st.write(
            f"**Internet:** {internet_service}"
        )


    # ========================================================
    # CUSTOMER SNAPSHOT
    # ========================================================

    st.divider()

    st.subheader(
        "Customer Snapshot"
    )

    snapshot1, snapshot2, snapshot3, snapshot4 = st.columns(4)

    with snapshot1:

        st.metric(
            "Tenure",
            f"{tenure} mo"
        )

    with snapshot2:

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.0f}"
        )

    with snapshot3:

        st.metric(
            "Contract",
            contract
        )

    with snapshot4:

        st.metric(
            "Internet",
            internet_service
        )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        "View Complete Customer Details"
    ):

        details = pd.DataFrame({

            "Customer Attribute": [

                "Gender",
                "Senior Citizen",
                "Partner",
                "Dependents",
                "Tenure",
                "Phone Service",
                "Multiple Lines",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Contract",
                "Paperless Billing",
                "Payment Method",
                "Monthly Charges",
                "Total Charges"

            ],

            "Selected Value": [

                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                monthly_charges,
                total_charges

            ]

        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "◈ ChurnIQ  •  Customer Churn Analysis  •  "
    "Powered by XGBoost"
)
