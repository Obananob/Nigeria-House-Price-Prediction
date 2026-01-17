import streamlit as st
import pandas as pd
import joblib


model = joblib.load('House_price.pkl')
model_columns = joblib.load('model_columns.pkl')


df = pd.read_csv('nigeria_houses_data.csv')


st.set_page_config(page_title="Naija House Predictor", layout="centered")
st.title(" Nigerian House Price Predictor")
st.caption("Powered by GB")


st.sidebar.header("House Details")


bedrooms = st.sidebar.slider("Bedrooms", 1, 9, 4)
bathrooms = st.sidebar.slider("Bathrooms", 1, 9, 4)
toilets = st.sidebar.slider("Toilets", 1, 9, 5)
parking_space = st.sidebar.slider("Parking Space", 1, 9, 4)



states = sorted(df['state'].unique())
selected_state = st.selectbox("State", states)


towns_in_state = sorted(df[df['state'] == selected_state]['town'].unique())
selected_town = st.selectbox("Town", towns_in_state)

types = sorted(df['title'].unique())
selected_title = st.selectbox("House Type", types)


if st.button("Predict Price", type="primary"):

    user_input = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'toilets': [toilets],
        'parking_space': [parking_space],
        'title': [selected_title],
        'town': [selected_town],
        'state': [selected_state]
    })



    user_input_encoded = pd.get_dummies(user_input)


    if 'model_columns' not in locals():
        
        model_columns = pd.get_dummies(df.drop('price', axis=1), columns=['title', 'town', 'state'], drop_first=True).columns

    final_input = user_input_encoded.reindex(columns=model_columns, fill_value=0)


    prediction = model.predict(final_input)[0]


    st.markdown("---")
    st.success(f"###  Estimated Value: ₦{prediction:,.2f}")


    st.info(f"Market Estimate for a {bedrooms}-bedroom {selected_title} in {selected_town}, {selected_state}.")
