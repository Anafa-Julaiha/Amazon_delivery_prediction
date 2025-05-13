import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")  # Replace with your actual model path

# Title
st.title("📦 Amazon Delivery Time Predictor")

st.markdown("Enter the order details to estimate delivery time.")

# Simple inputs
category = st.selectbox("Item Category", ["Electronics", "Clothing", "Groceries"])
area = st.selectbox("Delivery Area", ["Urban", "Suburban", "Rural"])
order_hour = st.slider("Order Hour (24H)", 0, 23, 14)
order_day = st.selectbox("Order Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
distance_km = st.number_input("Distance to Customer (km)", min_value=0.0, max_value=100.0, value=5.0)

# Encoding (example only — adjust based on your model's training)
category_map = {"Electronics": 0, "Clothing": 1, "Groceries": 2}
area_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}

category_encoded = category_map[category]
area_encoded = area_map[area]
order_day_encoded = day_map[order_day]

# Feature vector
features = np.array([[category_encoded, area_encoded, order_hour, order_day_encoded, distance_km]])

# Prediction
if st.button("Predict Delivery Time"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")

