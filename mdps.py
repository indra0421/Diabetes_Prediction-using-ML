# -*- coding: utf-8 -*-
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# converted '\' to '/' path
loaded_model = pickle.load(open("trained_model.sav","rb"))

# func for diabtetes prediction
def diabetes_prediction(arr):
    inp_data_as_nparray = np.asarray(arr)
    reshaped_data = inp_data_as_nparray.reshape(1,-1)
    # scaler.fit(reshaped_data)
    # std_data = scaler.transform(reshaped_data)
    predict = loaded_model.predict(reshaped_data)
    if(predict[0] == 0):
       return 'Non diabetic patient'
    else:
       return 'Diabetic patient'
   

# main func for streamlit -- user interface

def main():
    st.title('Diabetes Prediction Web App')
    
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.text_input('No of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('SkinThickness')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('Body mass index')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('Age')
    
    # code for prediction
    diagonosis = ''
    # btn creating and prediction
    if st.button('Diabetes Test Result'):
        diagonosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    # final output
    st.success(diagonosis)
    
    

if __name__ == '__main__':
    main()
    
    
    
    

