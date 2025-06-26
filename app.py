import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model


#loading model
model = load_model('model.h5')
#gender
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

#encoder 
with open('oh_encoder_geo.pkl','rb') as file:
    oh_encoder_geo = pickle.load(file)

#scaler 
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


#streamlit app

st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', oh_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    'Geography':[geography]
})
#onehot encode Geography column
geo_encoded = oh_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=oh_encoder_geo.get_feature_names_out(['Geography']))



input_data = pd.concat([input_data.drop(['Geography'],axis=1),geo_encoded_df],axis=1)
input_data_scaled = scaler.transform(input_data)
#prediction whether he is going to leave the bank or not
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(prediction_probability) 

if prediction_probability > 0.5 :
    st.write('This guy is likely to churn')
else :
    st.write('This guy do not churn')