import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("lg.pkl")

model = load_model()

st.title("Category Predictor")

user_text = st.text_input("Enter your text here:")

if st.button("Get Category"):
    if user_text.strip():
        prediction = model.predict([user_text])[0]
        
        st.write(f"Category: **{prediction}**")
    else:
        st.error("Please enter some text first.")
