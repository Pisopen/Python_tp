import streamlit as st
import pandas as pd
import sklearn
import joblib

user_input = st.text_input("Enter text :", "")
user_input = st.text_input("Enter text :", "")
user_input = st.text_input("Enter text :", "")

model = joblib.load("modelGame.pkl")

if (user_input) :
    Y_pred = model.predict_proba([user_input])[0]
    
    if Y_pred[0] > Y_pred[1] and Y_pred[0]> Y_pred[2]:
        st.text("hate_speech")
    elif Y_pred[1] > Y_pred[0] and Y_pred[1]> Y_pred[2] :
        st.text("offensive_language")
    else :
        st.text("neither")