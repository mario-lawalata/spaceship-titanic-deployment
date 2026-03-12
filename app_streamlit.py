import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load artifacts
# Pastikan kamu sudah menjalankan 'python pipeline.py' agar file ini terupdate
preprocessor = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title("ASG 04 MD - Mario Wilhelmus Lawalata - Spaceship Titanic Model Deployment")
    st.subheader("Passenger Details (13 Features)")
    
    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
        sleep = st.selectbox("CryoSleep", [True, False])
        dest = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        vip = st.selectbox("VIP Status", [True, False])
        deck = st.selectbox("Cabin Deck", ["B", "F", "A", "G", "E", "D", "C", "T"])
        side = st.selectbox("Cabin Side", ["P", "S"])
        age = st.slider("Age", 0, 100, 25)

    with col2:
        num = st.number_input("Cabin Number", 0, 2000, 0)
        rs = st.number_input("Room Service", 0.0, 10000.0, 0.0)
        fc = st.number_input("Food Court", 0.0, 10000.0, 0.0)
        sm = st.number_input("Shopping Mall", 0.0, 10000.0, 0.0)
        spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
        vr = st.number_input("VR Deck", 0.0, 10000.0, 0.0)

    # DataFrame Input (Urutan kolom harus SAMA dengan pre_processing.py)
    input_df = pd.DataFrame({
        'Age': [age], 'RoomService': [rs], 'FoodCourt': [fc], 'ShoppingMall': [sm],
        'Spa': [spa], 'VRDeck': [vr], 'Num': [num], 'HomePlanet': [home],
        'CryoSleep': [sleep], 'Destination': [dest], 'VIP': [vip], 'Deck': [deck], 'Side': [side]
    })

    if st.button("Click here to Predict"):
        # Jalankan preprocessing dan prediksi
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)
        
        st.divider()
        if prediction[0] == 1:
            st.success("RESULT: The passenger is predicted to be **TRANSPORTED** 🚀")
        else:
            st.error("RESULT: The passenger is predicted to be **NOT TRANSPORTED** 🛸")

if __name__ == "__main__":
    main()