import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')


st.title("Diabetes Prediction App")
st.write("This app predicts the likelihood of an individual developing diabetes")

model = load_model()


# User Inputs
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=5.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)

# Predict Price
if st.button("Predict Diabetes"):
    input_data = np.array([[bmi, HbA1c_level, blood_glucose_level]])
    prediction = model.predict(input_data)
    predicted_label = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"Prediction of Diabetes: {predicted_label}")
 
st.warning("This prediction is not scientifically proven and should not be used for medical diagnosis. It is based solely on the given parameters and a machine learning model.")