import streamlit as st
import joblib

@st.cache_resource
def load_ml():
    model = joblib.load("lg.pkl")
    vectorizer = joblib.load("vectorizer.pkl") 
    return model, vectorizer

model, vectorizer = load_ml()

st.title("Category Predictor")

user_text = st.text_input("Enter your text here:")

if st.button("Get Category"):
    if user_text.strip():
        transformed_text = vectorizer.transform([user_text])
        
        prediction = model.predict(transformed_text)[0]
        
        st.write(f"Category: **{prediction}**")
    else:
        st.error("Please enter some text first.")
