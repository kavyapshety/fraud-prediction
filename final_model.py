#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from PIL import Image
import joblib
import streamlit.components.v1 as components
from joblib import dump, load
import joblib

#load model
# load the model from disk
with open('fraudmodel.pkl','rb') as f:
        model=joblib.load(f)

    
def user_input_features(step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    step=float(step)
    amount=float(amount)
    oldbalanceOrg=float(oldbalanceOrg)
    newbalanceOrig=float(newbalanceOrig)
    oldbalanceDest=float(oldbalanceDest)
    newbalanceDest=float(newbalanceDest)
    if type == 'PAYMENT':
        type = 0
    elif type == 'TRANSFER':
        type = 1
    elif type == 'CASH_OUT':
        type = 2
    elif type == 'DEBIT':
        type = 3
    elif type == 'CASH_IN':
        type = 4    
    else:
        print(ValueError)
    arr_val=np.array([step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]).reshape(1,-1)
    prediction = model.predict(arr_val)
    
    if prediction == 0:
        output ="Not_Fraud"
    else:
        output ="Fraud"   
    return output


def main():
    # set page title
    st.set_page_config('Fraud Detection')

    st.title('Fraud Transaction Prediction System')
    # Using "with" notation
    with st.sidebar:
        image= Image.open("iidt_logo_137.jpeg")
        add_image=st.image(image,use_column_width=True)

    social_acc = ['About']
    social_acc_nav = st.sidebar.selectbox('About', social_acc)
    if social_acc_nav == 'About':
        st.sidebar.markdown("<h2 style='text-align: center;'> This Project completed under ExcelR, the team completed the project:</h2> ", unsafe_allow_html=True)
        st.sidebar.markdown('''---''')
        st.sidebar.markdown('''
        • Miss. Kavya M P \n
        • Mr. Raju A S \n
        • Mr. AMAN SAJAD \n 
        • Mr.Raghavendra R \n
        ''')
        st.sidebar.markdown("[ Visit To Github Repositories](https://github.com/kavyapshety/fraud-prediction.git)")   
    menu_list = ["FRAUD PREDICTION"]
    menu = st.radio("Menu", menu_list)
               
    if menu == 'FRAUD PREDICTION':
            
            st.title("FRAUD PREDICTION")
            #import the image
            image= Image.open("Header_Fraud_Detection_and_Prevention.jpeg")
            st.image(image,use_column_width=True)

            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Fraud Transaction Prediction </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)
            #step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
            step = st.text_input("Step","Type Here")
            amount = st.text_input("Amount Transfer","Type Here")
            oldbalanceOrg = st.text_input("Sender Old Balance","Type Here")
            newbalanceOrig = st.text_input("Sender New Balance","Type Here")
            oldbalanceDest = st.text_input("Receiver Old Balance","Type Here")
            newbalanceDest = st.text_input("Receiver New Balance","Type Here")
            type = st.selectbox('Payment Type', ('CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_IN', 'DEBIT'))
            result=""
            if st.button("Predict"):
                result=user_input_features(step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
            st.success('The predicted Annual spent is {}'.format(result))
            if st.button("About"):
                st.text("The aim of this project is to predict that the transaction is Fraud or Not")
                st.text("based on their values and transaction type.")
                st.text("ExcelR Project")
            
if __name__=='__main__':
    main()
    
    

