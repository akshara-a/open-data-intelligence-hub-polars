import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="HR Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    height: 3em;
    border-radius: 10px;
    border: none;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff2e2e;
}

[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #333;
}

.css-1d391kg {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load("hr_model.pkl")

    return model

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Project Info")

st.sidebar.markdown("""
## HR Analytics AI

AI-powered employee attrition prediction system.

### 🔧 Technologies
- Python
- Scikit-Learn
- Streamlit
- Pandas
- Plotly

### 🤖 ML Algorithm
KNeighborsClassifier

### 📊 Dataset
HR Analytics Dataset

### 👨‍💻 Developed By
SANJANA MANTHENA
""")

# =====================================================
# HEADER
# =====================================================

st.title("📊 HR Analytics Attrition Prediction")

st.markdown("""
Predict whether an employee is likely to leave the company using Machine Learning.
""")

st.divider()

# =====================================================
# TOP METRICS
# =====================================================

# m1, m2, m3, m4 = st.columns(4)

# m1.metric("Model", "KNN")
# m2.metric("Dataset", "HR Analytics")
# m3.metric("Algorithm", "Classification")
# m4.metric("Deployment", "Streamlit")

# st.divider()

# =====================================================
# LAYOUT
# =====================================================

left_col, right_col = st.columns([1, 1])

# =====================================================
# INPUT SECTION
# =====================================================

with left_col:

    st.subheader("🧑 Employee Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            'Travel_Rarely',
            'Travel_Frequently',
            'Non-Travel'
        ]
    )

    department = st.selectbox(
        "Department",
        [
            'Sales',
            'Research & Development',
            'Human Resources'
        ]
    )

    education_field = st.selectbox(
        "Education Field",
        [
            'Life Sciences',
            'Medical',
            'Marketing',
            'Technical Degree',
            'Human Resources',
            'Other'
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            'Male',
            'Female'
        ]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            'Sales Executive',
            'Research Scientist',
            'Laboratory Technician',
            'Manufacturing Director',
            'Healthcare Representative',
            'Manager',
            'Sales Representative',
            'Research Director',
            'Human Resources'
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        [
            'Yes',
            'No'
        ]
    )

# =====================================================
# SECOND COLUMN
# =====================================================

with right_col:

    st.subheader("💼 Work Information")

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=5
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=3
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🚀 Predict Attrition")

# =====================================================
# PREDICTION
# =====================================================

if predict_btn:

    input_df = pd.DataFrame({

        'Age': [age],
        'BusinessTravel': [business_travel],
        'Department': [department],
        'EducationField': [education_field],
        'Gender': [gender],
        'JobRole': [job_role],
        'MonthlyIncome': [monthly_income],
        'OverTime': [overtime],
        'TotalWorkingYears': [total_working_years],
        'YearsAtCompany': [years_at_company],
        'JobSatisfaction': [job_satisfaction],
        'WorkLifeBalance': [work_life_balance]

    })

    with st.spinner("🔍 Analyzing employee data..."):

        time.sleep(2)

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

    attrition_prob = probability[1] * 100

    st.divider()

    # =====================================================
    # RESULT SECTION
    # =====================================================

    r1, r2 = st.columns([1, 1])

    # =====================================================
    # LEFT RESULT
    # =====================================================

    with r1:

        st.subheader("📌 Prediction Result")

        if prediction == 1:

            st.error("⚠️ HIGH ATTRITION RISK")

            st.markdown(f"""
            ### Employee may leave the company

            #### Attrition Probability:
            ## {attrition_prob:.2f}%
            """)

        else:

            st.success("✅ LOW ATTRITION RISK")

            st.markdown(f"""
            ### Employee likely to stay

            #### Retention Probability:
            ## {(100 - attrition_prob):.2f}%
            """)

    # =====================================================
    # RIGHT RESULT - GAUGE
    # =====================================================

    with r2:

        st.subheader("📊 Attrition Risk Meter")

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=attrition_prob,

            title={'text': "Attrition Risk %"},

            gauge={

                'axis': {'range': [0, 100]},

                'bar': {'color': "red"},

                'steps': [

                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}

                ],
            }
        ))

        fig.update_layout(
            height=350,
            paper_bgcolor="#0E1117",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =====================================================
    # AI INSIGHTS
    # =====================================================

    st.subheader("🤖 AI Insights")

    insights = []

    if overtime == "Yes":
        insights.append("• Employee works overtime frequently.")

    if job_satisfaction <= 2:
        insights.append("• Job satisfaction is low.")

    if work_life_balance <= 2:
        insights.append("• Work-life balance is poor.")

    if monthly_income < 3000:
        insights.append("• Employee income is relatively low.")

    if years_at_company < 2:
        insights.append("• Employee is relatively new to the company.")

    if len(insights) == 0:
        insights.append("• Employee profile looks stable.")

    for i in insights:
        st.info(i)

    st.divider()

    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader("📄 Employee Data Summary")

    st.dataframe(input_df, use_container_width=True)