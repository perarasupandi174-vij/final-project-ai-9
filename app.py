import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Customer Support Ticket Classification")

text = st.text_area("Enter your support ticket")

if st.button("Predict"):
    if text:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)
        st.success(f"Issue Type: {prediction[0]}")
    else:
        st.warning("Please enter some text")
