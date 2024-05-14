import pandas as pd
import numpy as np
import streamlit as st
import pickle
import sklearn

st.title("Airline Passenger Satisfaction")


model=pickle.load(open('classifier.pkl','rb'))
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def user_data():
    gender=st.radio("Select Gender of Passenger : ",options=['Male','Female'])
    c_type=st.radio("Select Customer Type : ",options=['Loyal customer','Disloyal customer'])
    age=st.number_input(min_value=18,max_value=110,value=None,placeholder="Enter valid age",label="Actual age of the passengers")
    t_type=st.radio("Type of Travel",options=['Personal Travel', 'Business Travel'])
    cls=st.radio("Travel class in the plane of the passengers",options=["Business", "Eco", "Eco Plus"])
    dt=st.number_input(min_value=2.7,max_value=15715.0,step=0.1,value=None,placeholder="Enter valid distance (in Km)",label="Flight distance of this journey")
    wifi=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of the inflight wifi service")
    time_conv=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of Departure/Arrival time convenient")
    gate=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of Gate location")
    food=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of  Food and drink")
    boarding=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of online boarding")
    seat=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of  Seat comfort")
    on_board=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of On-board servicen")
    leg_room=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of Leg room service")
    baggage=st.select_slider(options=[1,2,3,4,5],label="Satisfaction level of baggage handling")
    checkin=st.select_slider(options=[0,1,2,3,4,5],label="Satisfaction level of Check-in service")
    departure_delay=st.number_input(min_value=0,max_value=1700,value=None,placeholder="Enter departure delay in Minutes",label="Minutes delayed when departure")
    
    encod={'Male':1 ,'Female':0,'Loyal customer':0 ,'Disloyal customer':1,'Personal Travel':1, 'Business Travel':0,'Eco Plus':2 ,'Business':0 ,'Eco':1}
    gender_dt=encod[gender]
    c_type_dt=encod[c_type]
    t_type_dt=encod[t_type]
    cls_dt=encod[cls]
    if age!=None and dt!=None and departure_delay!=None:
        user_data_fed={
            'Gender':gender_dt,
            'Customer Type':c_type_dt,
            'Age':age,
            'Type of Travel':t_type_dt,
            'Class':cls_dt,
            'Flight Distance':dt,
            'Inflight wifi service':wifi,
            'Departure/Arrival time convenient':time_conv,
            'Gate location':gate,
            'Food and drink':food,
            'Online boarding':boarding,
            'Seat comfort':seat,
            'On-board service':on_board,
            'Leg room service':leg_room,
            'Baggage handling':baggage,
            'Checkin service':checkin,
            'Departure Delay in Minutes':departure_delay
        }
        report=pd.DataFrame(user_data_fed,index=[0])
        return report
    else:
        st.warning("Fill all the data")
        return "An Error Occured"
choice=st.sidebar.radio("Menu",["Prediction"])
if choice:
    client=user_data()
    if type(client)==str:
        st.warning(client)
    else:
        #st.write(client)
        scaling_need=['Age','Flight Distance','Departure Delay in Minutes']
        #st.write(type(client))
        client[scaling_need]=scaler.transform(client[scaling_need])
        #st.write(client)
        val=model.predict(client)
        #st.write(val)
        if val==0:
            st.subheader('Customer is Neutral or Dissatisfied about the service')
        else:
            st.subheader('Customer is satisfied about the service')