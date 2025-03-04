import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("my_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Predicción de Valores de Vivienda")

st.write("Ingresa las características de la vivienda:")

longitude = st.number_input("Longitud", value=-118.0)
latitude = st.number_input("Latitud", value=34.0)
housing_median_age = st.number_input("Edad Media de la Vivienda", value=30)
total_rooms = st.number_input("Número Total de Habitaciones", value=2000)
total_bedrooms = st.number_input("Número Total de Dormitorios", value=400)
population = st.number_input("Población", value=800)
households = st.number_input("Número de Hogares", value=400)
median_income = st.number_input("Ingreso Medio", value=3.0)
ocean_proximity = st.selectbox("Proximidad al Océano", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                            population, households, median_income, ocean_proximity]],
                          columns=["longitude", "latitude", "housing_median_age", "total_rooms",
                                   "total_bedrooms", "population", "households", "median_income",
                                   "ocean_proximity"])

input_data_prepared = preprocessor.transform(input_data)

if st.button("Predecir Precio de la Vivienda"):
    prediction = model.predict(input_data_prepared)
    st.write(f"**Precio Predicho:** ${prediction[0]:,.2f}")
