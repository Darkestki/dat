import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
import time

# ------------------------------------------------------
# 🎉 Custom Gym Balloons Animation
# ------------------------------------------------------
def gym_balloons():
    gym_emojis = ["🏋️‍♂️", "💪", "🔥", "🏃‍♂️", "✨"]
    placeholder = st.empty()

    for i in range(30):  # number of emoji bursts
        emoji = gym_emojis[i % len(gym_emojis)]
        placeholder.markdown(
            f"<h1 style='text-align:center; font-size:60px;'>{emoji}</h1>",
            unsafe_allow_html=True
        )
        time.sleep(0.07)
    placeholder.empty()

# ------------------------------------------------------
# 🧠 Load Model + Scaler
# ------------------------------------------------------
model_filename = "calorie_svc_model.pkl"
scaler_filename = "scaler.pkl"

try:
    loaded_model = joblib.load(open(model_filename, "rb"))
    scaler = joblib.load(open(scaler_filename, "rb"))
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# ------------------------------------------------------
# ⚙️ Streamlit Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="🔥 Gym Calorie Predictor", page_icon="🏋️‍♂️", layout="centered")
st.title("🏋️‍♀️ Gym Member Calorie Burn Prediction")
st.markdown("Enter your workout details below to get an **estimated calorie burn**.")
st.divider()

# ------------------------------------------------------
# 📥 Input Section
# ------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("🧍 Age", 10, 80, 25)
    Gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    Weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 70.0)
    Height = st.number_input("📏 Height (m)", 1.0, 2.5, 1.70)
    Max_BPM = st.number_input("❤️ Max Heart Rate", 60, 220, 180)
    Avg_BPM = st.number_input("💓 Avg Heart Rate", 40, 200, 130)
    Resting_BPM = st.number_input("🫀 Resting BPM", 40, 120, 70)

with col2:
    Session_Duration = st.number_input("⏱️ Session Duration (hrs)", 0.1, 5.0, 1.0)
    Fat_Percentage = st.number_input("💪 Body Fat %", 5.0, 60.0, 20.0)
    Water_Intake = st.number_input("💧 Water Intake (L)", 0.0, 5.0, 1.5)
    Workout_Frequency = st.number_input("📅 Workout Frequency (days/week)", 1, 7, 4)
    Experience_Level = st.selectbox("🎯 Experience Level", ["Beginner", "Intermediate", "Advanced"])
    BMI = round(Weight / (Height ** 2), 2)
    Workout_Type = st.selectbox("🏃 Workout Type", ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"])

# ------------------------------------------------------
# 🧩 Encoding
# ------------------------------------------------------
exp_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
gender_map = {"Male": 1, "Female": 0, "Other": 2}

input_dict = {
    "Age": Age,
    "Weight (kg)": Weight,
    "Height (m)": Height,
    "Max_BPM": Max_BPM,
    "Avg_BPM": Avg_BPM,
    "Resting_BPM": Resting_BPM,
    "Session_Duration (hours)": Session_Duration,
    "Fat_Percentage": Fat_Percentage,
    "Water_Intake (liters)": Water_Intake,
    "Workout_Frequency (days/week)": Workout_Frequency,
    "Experience_Level": exp_map[Experience_Level],
    "BMI": BMI,
    "Gender_Encoded": gender_map[Gender],
    "Workout_Type_Encoded": ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"].index(Workout_Type),
}

input_df = pd.DataFrame([input_dict])

# ------------------------------------------------------
# 🔍 Prediction
# ------------------------------------------------------
st.divider()
if st.button("🔥 Predict Calories Burned"):
    try:
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict using SVR / SVC regression model
        prediction = loaded_model.predict(scaled_input)

        # Convert to float
        calories = float(prediction[0])

        st.success(f"🔥 **Predicted Calories Burned:** {round(calories, 2)} kcal")

        # Custom gym celebration
        gym_balloons()

        st.markdown("Keep pushing your limits! 💪🔥")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
        expected = getattr(loaded_model, 'feature_names_in_', None)
        if expected is not None:
            st.write("Model expects:", list(expected))
        st.write("Your input:", list(input_df.columns))

# ------------------------------------------------------
# ❤️ Footer
# ------------------------------------------------------
st.divider()
st.caption("Developed with ❤️ using Streamlit & ML 🏋️‍♂️🔥")
