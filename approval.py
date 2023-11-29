# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:00:27 2023

@author: saieeh
"""

# Save this code in a file, e.g., loan_approval_app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model
model_filename = 'final_model.pkl'  # Replace with your actual model filename
with open(model_filename, 'rb') as model_file:
    loan_model = pickle.load(model_file)

# Streamlit app
def main():
    st.title("Loan Approval App")

    # Input features
    st.sidebar.header("User Input Features")

    # Example input features (replace with your actual features)
    Age = st.sidebar.slider("Age", 19, 50)
    Experience = st.sidebar.slider("Experience", 1, 50)
    CCAvg = st.sidebar.slider("Cash Credit", 1, 50)
    income = st.sidebar.slider("Annual Income (in Thousand)",  1, 500)
    Mortgage = st.sidebar.slider("Mortgage Loan ", 1, 2000)
    Education = st.sidebar.selectbox("Education ", [0, 1, 2])
    CD_account = st.sidebar.selectbox("CD Account ", [0, 1])
    CreditCard = st.sidebar.selectbox("Credit Card (1 for have a card)", [0, 1])
    
    # Convert input features to a NumPy array
    features = np.array([[Age, Experience, CCAvg, income, CD_account, Mortgage, Education, CreditCard]])

    # Display the input features
    st.write("## Input Features")
    st.write(f"Age: {Age}")
    st.write(f"Experience: {Experience}")
    st.write(f"Cash Credit: {CCAvg}")
    st.write(f"Annual Income: {income}")
    st.write(f"Mortgage: {Mortgage}")
    st.write(f"Education: {Education}")
    st.write(f"CD Account: {CD_account}")
    st.write(f"Credit Card: {CreditCard} ")

    # Loan approval prediction
    if st.button("Predict Loan Approval"):
        prediction = loan_model.predict(features)

        if prediction[0] == 1:
            st.success("Congratulations! Your loan is approved.")
        else:
            st.error("Sorry, your loan application is not approved.")

if __name__ == "__main__":
    main()
