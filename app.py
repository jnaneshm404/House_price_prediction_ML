import streamlit as st
import joblib
import numpy as np
import pandas as pd

model= joblib.load("house_price_model.pkl")
le=joblib.load("locations.pkl")

st.title("Bengaluru House Price Predictor")
st.write("Predict house price in Lakhs (based  on realistic Bengaluru data)")

col1, col2=st.columns(2)

with col1:
    area=st.number_input("Area (sq ft)", 600, 3000, 1200)
    bedrooms = st.number_input("Bedrooms", 1,5,2)
    bathrooms=st.number_input("Bathrooms",1,4,2)

with col2:
    location=st.selectbox("Location", sorted(le.classes_))
    age=st.number_input("Age of Building (years)",0,30,5)
    parking=st.selectbox("Parking", [0,1])

if st.button("Predict Price"):
    loc=le.transform([location])[0]
    features=np.array([[area, bedrooms, bathrooms, loc,age,parking]])
    price=model.predict(features)[0]
    st.success(f"Predicted Price: **Rs. {price:.1f} Lakhs**")
    st.info(f"Approximate: Rs. {price * 100000:,.0f}")