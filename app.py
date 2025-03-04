import streamlit as st
import numpy as np
import pandas as pd
import joblib

class CombinedAttributesAdder:
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def main():
    st.title('Housing Price Predictor')
    
    try:
        full_pipeline = joblib.load('models/full_pipeline.joblib')
        forest_reg = joblib.load('models/final_model.joblib')
        st.sidebar.success('Model loaded successfully.')
    except FileNotFoundError:
        st.sidebar.error('Model files not found. Please ensure models are in the correct directory.')
        return

    st.header('Enter Housing Details')
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=-122.0)
        latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=37.0)
        housing_median_age = st.number_input('Housing Median Age', min_value=0, max_value=100, value=30)
        total_rooms = st.number_input('Total Rooms', min_value=0, value=2000)
    
    with col2:
        total_bedrooms = st.number_input('Total Bedrooms', min_value=0, value=400)
        population = st.number_input('Population', min_value=0, value=1000)
        households = st.number_input('Households', min_value=0, value=400)
        median_income = st.number_input('Median Income', min_value=0.0, value=3.0, step=0.1)
    
    ocean_proximity = st.selectbox('Ocean Proximity', 
        ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    )
    
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    if st.button('Predict House Value'):
        try:
            input_prepared = full_pipeline.transform(input_data)
            
            prediction = forest_reg.predict(input_prepared)[0]
            
            st.success(f'Predicted House Value: ${prediction:,.2f}')
        except Exception as e:
            st.error(f'Error in prediction: {e}')

if __name__ == '__main__':
    main()