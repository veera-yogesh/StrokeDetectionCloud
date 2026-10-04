
import streamlit as st
from resource import load_resources, predict_stroke

st.set_page_config(
    page_title="Stroke Prediction Demo",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def initialize_model():
    load_resources()
    return True

st.title("🧠 Stroke Prediction Demo")
st.write(
    "Enter the requested details to see the output "
    "of the machine learning model."
)

st.warning(
    "Educational project only. This model is not "
    "medically validated and cannot diagnose or "
    "reliably assess an individual's stroke risk."
)

try:
    initialize_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

st.subheader("Enter Input Details")

with st.form("prediction_form"):
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=45,
        step=1
    )

    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        max_value=500.0,
        value=110.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

    submitted = st.form_submit_button(
        "Run Model"
    )

if submitted:
    values = [
        age,
        hypertension,
        heart_disease,
        avg_glucose_level,
        bmi
    ]

    try:
        result = predict_stroke(values)

        st.subheader("Model Output")

        if result["prediction"] == 1:
            st.error("Positive model classification")
        else:
            st.info("Negative model classification")

        st.metric(
            "Model output",
            f'{result["probability"] * 100:.2f}%'
        )

        st.caption(
            "This percentage is the model's output, "
            "not a medically validated stroke probability."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
