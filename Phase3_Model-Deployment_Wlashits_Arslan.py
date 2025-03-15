import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Load the model from MLflow
MODEL_URI = "models:/Best_Model/3"  # Update with actual model version
model = mlflow.pyfunc.load_model(MODEL_URI)

# Streamlit UI
st.title("Company Profit Prediction App")
st.write("Enter the feature values below to get a profit prediction.")

# Feature inputs
rd_spend = st.slider("R&D Spend ($)", 0, 1000000, 500000)
administration = st.slider("Administration Cost ($)", 0, 500000, 200000)
marketing_spend = st.slider("Marketing Spend ($)", 0, 1000000, 300000)
state = st.selectbox("State", ["New York", "Florida"])  # Removed California since it was not seen during training

# One-hot encoding for State
state_features = {"State_Florida": 0, "State_New York": 0}  # Only the states seen during training
if f"State_{state}" in state_features:
    state_features[f"State_{state}"] = 1  # Set the selected state to 1

# Convert inputs to DataFrame
features = pd.DataFrame({
    "R&D Spend": [rd_spend],
    "Administration": [administration],
    "Marketing Spend": [marketing_spend],
    **state_features  # Add one-hot encoded state features
})

# Predict button
if st.button("Predict Profit"):
    prediction = model.predict(features)
    
    # Assuming the model provides uncertainty estimates
    pred_mean = prediction[0]
    pred_std = np.std(prediction) if len(prediction) > 1 else 5000  # Adjust std value
    lower_bound = pred_mean - 1.96 * pred_std
    upper_bound = pred_mean + 1.96 * pred_std
    
    st.success(f"Predicted Profit: ${pred_mean:,.2f} [95% CI: ${lower_bound:,.2f} - ${upper_bound:,.2f}]")
