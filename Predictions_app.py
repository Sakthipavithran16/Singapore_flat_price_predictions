import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import pickle

df = pd.read_csv(r"D:/Singapore_resale_flat_predictions/Final_flat_resale_df.csv")

st.set_page_config(layout="wide")

st.title("Singapore Resale Flat Prices Predictions")

option = option_menu(None,options = ["About","Flat Prices Predictions"],
                       icons = ["house-door","activity"],
                       default_index=0,
                       orientation="horizontal", 
                       styles={"nav-link-selected": {"background-color": "#e20afc"}})



town_options  = list(df['town'].unique())
flat_type_options  = list(df['flat_type'].unique())
street_name_options  = list(df['street_name'].unique())
flat_model_options  = list(df['flat_model'].unique())



if option == "About":

    st.write("")

    st.write('''The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate 
             the resale value of a flat.''')
    
    st.write('''There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration.''')

    st.write('''A predictive model can help to overcome these challenges by providing users with an estimated resale price based 
             on these factors.''')
    
    st.write("")

    st.write("")

    st.write('''This web application predicts the resale prices of flats in Singapore.''')

    st.write('''This predictive model will be based on historical data of resale flat transactions, 
                and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.''')
    

elif option == "Flat Prices Predictions":

    with st.form('Flat resale'):

        col1, col2, col3 = st.columns([5, 2, 5])

        with col1:
            st.write(' ')
            block = st.text_input('Block (Min: 1, Max: 980)')
            lease_commence_date = st.text_input('Lease Commence Date (Min : 1966 , Max : 2018)')
            resale_year = st.selectbox('Resale Year', sorted(list(range(1990,2025))))
            resale_month = st.selectbox('Resale Month', sorted(list(range(1,13))))
            storey_lower_limit = st.selectbox('Storey Lower Limit', sorted(list(range(1,20))))
            storey_upper_limit = st.selectbox('Storey Upper Limit', sorted(list(range(3,22))))

        with col3:
            st.write()
            floor_area_sqm = st.text_input('Floor Area Sqm (Min : 28 , Max : 173)')
            town = st.selectbox('Town', sorted(town_options))
            flat_type = st.selectbox('Flat Type', sorted(flat_type_options))
            street_name = st.selectbox('Street Name', sorted(street_name_options))
            flat_model = st.selectbox('Flat Model', sorted(flat_model_options))

        st.write('')
        submit_button = st.form_submit_button(label='SUBMIT')


    if submit_button :

        with open('town_encoding.pkl', 'rb') as f:
            town_encoding = pickle.load(f)

        with open('street_name_encoding.pkl', 'rb') as f:
            street_name_encoding = pickle.load(f)

        with open('flat_model_encoding.pkl', 'rb') as f:
            flat_model_encoding = pickle.load(f)
        
        with open('flat_type_encoding.pkl', 'rb') as f:
            flat_type_encoder = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('xgb_regressor_model.pkl', 'rb') as f:
            regression_model = pickle.load(f)
        
        storey_average = (int(storey_lower_limit) + int(storey_upper_limit)) / 2 


        new_data = np.array([[float(block),float(floor_area_sqm),float(lease_commence_date),int(resale_year),int(resale_month),
                              float(storey_average),town,street_name,flat_model,flat_type]])
        
        new_data_le1 = town_encoding.transform(new_data[:,[6]])
        new_data_le2 = street_name_encoding.transform(new_data[:,[7]])
        new_data_le3 = flat_model_encoding.transform(new_data[:,[8]])

        flat_type_from_new_data = new_data[0, 9]
        flat_type_encoded = flat_type_encoder.get(flat_type_from_new_data, None)

        new_data = np.concatenate((new_data[:, [0,1,2,3,4,5]], new_data_le1.reshape(-1, 1),new_data_le2.reshape(-1, 1),
                                   new_data_le3.reshape(-1, 1),np.array([[flat_type_encoded]])), axis=1)

        new_data = scaler.transform(new_data)
        new_pred = regression_model.predict(new_data)
        
        rounded_prediction = np.round(new_pred[0], 2)

        st.write(f'## :green[Predicted Flat resale price: {rounded_prediction:.2f}]')
