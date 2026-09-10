"""
Student Performance Prediction — Simple Streamlit App
--------------------------------------------------------
This app just LOADS the model we already trained and saved in notebook.ipynb
(best_model.pkl) and uses it to make predictions. It does not train anything.

To run this app:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write(
    "Fill in a student's info below, and this app will predict their "
    "likely final grade (A, B, C, D, or F) using a machine learning "
    "model trained on past student data."
)

MODEL_FILE = "best_model.pkl"

# ---- Step 1: Load the saved model ----
if not os.path.exists(MODEL_FILE):
    st.error(
        f"Could not find '{MODEL_FILE}'. Please run notebook.ipynb first — "
        "it trains the model and saves this file — then put it in the same "
        "folder as app.py."
    )
    st.stop()

saved = joblib.load(MODEL_FILE)
pipeline = saved["pipeline"]
label_encoder = saved["label_encoder"]
info = saved["info"]

# ---- Step 2: Build the input form ----
st.header("Student Information")

gender = st.selectbox("Gender", info["category_options"]["gender"])

study_min, study_max, study_avg = info["numeric_ranges"]["study_time_hours"]
study_time_hours = st.slider("Study time per day (hours)", 0.0, 12.0, round(study_avg, 1))

att_min, att_max, att_avg = info["numeric_ranges"]["attendance_percent"]
attendance_percent = st.slider("Attendance (%)", 0.0, 100.0, round(att_avg, 1))

sleep_min, sleep_max, sleep_avg = info["numeric_ranges"]["sleep_hours"]
sleep_hours = st.slider("Sleep per night (hours)", 0.0, 12.0, round(sleep_avg, 1))

parental_education = st.selectbox("Parental education", info["category_options"]["parental_education"])

internet_access = st.radio("Internet access at home?", info["category_options"]["internet_access"])

extracurricular_activities = st.radio("Does extracurricular activities?", info["category_options"]["extracurricular_activities"])

part_time_job = st.radio("Has a part-time job?", info["category_options"]["part_time_job"])

prev_min, prev_max, prev_avg = info["numeric_ranges"]["previous_grade"]
previous_grade = st.slider("Previous grade (0-100)", 0.0, 100.0, round(prev_avg, 1))

# ---- Step 3: Predict button ----
if st.button("Predict Final Grade"):
    try:
        # Put the inputs into a table (DataFrame) with the same column names
        # the model was trained on
        student_data = pd.DataFrame([{
            "gender": gender,
            "study_time_hours": study_time_hours,
            "attendance_percent": attendance_percent,
            "sleep_hours": sleep_hours,
            "parental_education": parental_education,
            "internet_access": internet_access,
            "extracurricular_activities": extracurricular_activities,
            "part_time_job": part_time_job,
            "previous_grade": previous_grade,
        }])

        # The pipeline handles all the scaling/encoding by itself
        prediction_number = pipeline.predict(student_data)[0]
        prediction_grade = label_encoder.inverse_transform([prediction_number])[0]

        st.success(f"Predicted Final Grade: **{prediction_grade}**")

        # Show probability for each grade, if the model supports it
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(student_data)[0]
            prob_table = pd.DataFrame({
                "Grade": label_encoder.classes_,
                "Probability": probabilities,
            }).sort_values("Grade")
            st.write("Probability for each grade:")
            st.bar_chart(prob_table.set_index("Grade"))

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ---- Step 4: Simple model info section ----
with st.expander("About this model"):
    st.write(f"**Model used:** {info['best_model_name']}")
    st.write(f"**Possible grades:** {', '.join(info['grade_names'])}")
    st.write(
        "Note: this model does not use the final exam score as an input "
        "— it predicts the grade from study habits and background info alone."
    )
