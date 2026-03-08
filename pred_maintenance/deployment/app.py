import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rishugupta92/predictive-maintenance",filename="predictive_maintenance_model_v1_0.joblib")

#Load the trained model
model = joblib.load(model_path)

# Streamlit UI for Purchase Prediction
st.title("Engine Failure Prediction App")
st.write("""
This application predicts the likelihood of a engine failure for timely maintenance.
Please enter the operating details below.
""")

# Operating Details
Engine_rpm = st.number_input("Engine rpm", min_value=60, max_value=2500, value=800)
Lub_oil_pressure = st.number_input("Lub oil pressure", min_value=60, max_value=2500, value=800)
Fuel_pressure = st.number_input("Fuel pressure", min_value=60, max_value=2500, value=800)
Coolant_pressure = st.number_input("Coolant pressure", min_value=60, max_value=2500, value=800)
lub_oil_temp = st.number_input("lub oil temp", min_value=60, max_value=2500, value=800)
Coolant_temp = st.number_input("Coolant temp", min_value=60, max_value=2500, value=800)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_rpm,
    'Lub oil pressure': Lub_oil_pressure,
    'Fuel pressure': Fuel_pressure,
    'Coolant pressure': Coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': Coolant_temp,
}])


# Classification threshold
classification_threshold = 0.5

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    if prediction == 1:
        st.success("❌ The engine is likely to fail, need immediate maintenance.")
    else:
        st.error("✅ The engine is likely to opearte without failure.")
