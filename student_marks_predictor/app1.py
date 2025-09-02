import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and data
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("student_marks_data.csv")

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Marks Predictor")

st.write("Enter the number of hours studied to predict the expected marks:")

# Input
hours = st.slider("Hours Studied", 0.0, 10.0, 1.0, 0.1)

# Predict
if st.button("Predict"):
    input_data = [[hours]]
    predicted_marks = model.predict(input_data)[0]
    st.success(f"📘 Predicted Marks: {predicted_marks:.2f}")

# Sidebar - Model Info
st.sidebar.header("📈 Model Information")
st.sidebar.markdown("- Model: Linear Regression")
st.sidebar.markdown("- R² Score: **0.89**")
st.sidebar.markdown("- Trained on 1000 entries")

# Show data sample
if st.checkbox("🧾 Show Sample Dataset"):
    st.dataframe(df.head())

# Chart: Hours vs Marks
st.subheader("📊 Training Data Overview")
st.line_chart(df.sort_values("Hours_Studied"))
