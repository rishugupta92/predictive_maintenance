%%writefile pred_maintenance/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="rishugupta92/predictive-maintenance-model",
    filename="predictive_maintenance_model_v1_0.joblib",
    repo_type="model"
)

# Load the trained model
model = joblib.load(model_path)

st.title("Engine Failure Prediction App")
st.write("""
This application predicts the likelihood of engine failure for timely maintenance.
Please enter the operating details below.
""")

Engine_rpm = st.number_input("Engine rpm", min_value=60, max_value=2500, value=800)
Lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0, max_value=8.0, value=3.3)
Fuel_pressure = st.number_input("Fuel pressure", min_value=0.0, max_value=22.0, value=6.7)
Coolant_pressure = st.number_input("Coolant pressure", min_value=0.0, max_value=8.0, value=2.3)
lub_oil_temp = st.number_input("lub oil temp", min_value=65.0, max_value=100.0, value=78.0)
Coolant_temp = st.number_input("Coolant temp", min_value=55.0, max_value=200.0, value=78.0)

input_data = pd.DataFrame([{
    "Engine rpm": Engine_rpm,
    "Lub oil pressure": Lub_oil_pressure,
    "Fuel pressure": Fuel_pressure,
    "Coolant pressure": Coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": Coolant_temp,
}])

classification_threshold = 0.5

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        st.success("❌ The engine is likely to fail, need immediate maintenance.")
    else:
        st.error("✅ The engine is likely to opearte without failure.")
