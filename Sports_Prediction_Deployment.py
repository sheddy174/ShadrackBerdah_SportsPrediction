import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
voting_regressor = joblib.load('voting_regressor.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Player Rating Prediction")

# Define input fields for user to enter new data
st.header("Enter player attributes:")

# Example of features (add or modify based on your model's features)
age = st.number_input('Age', min_value=16, max_value=45, value=25)
height = st.number_input('Height (cm)', min_value=150, max_value=210, value=180)
weight = st.number_input('Weight (kg)', min_value=50, max_value=120, value=75)
# Add more inputs as per your model's requirements

# Collect inputs into a single array
input_data = np.array([[age, height, weight]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    # Make prediction
    prediction = voting_regressor.predict(input_data_scaled)

    # Display the prediction and confidence score
    st.subheader("Predicted Rating")
    st.write(f"{prediction[0]:.2f}")
    
    # Confidence score (if applicable)
    st.subheader("Confidence Score")
    confidence_score = np.max(voting_regressor.predict_proba(input_data_scaled))  # Example for classifier
    st.write(f"{confidence_score:.2f}")
