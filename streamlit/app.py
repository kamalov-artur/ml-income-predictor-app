import streamlit as st
import pandas as pd
import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_URL = 'https://api.artkmlv.ru/'


categories = requests.get(API_URL + "/person_categories").json()

st.title("🔮 Прогноз уровня дохода")

st.markdown("""
Это приложение использует модель машинного обучения, чтобы предсказать, превышает ли годовой доход человека **50,000$**.

Чтобы получить предсказание, заполните данные ниже.  
Модель обучена на датасете [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income/data) и учитывает социально-демографические характеристики.
""")


with st.container():

    col1, col2 = st.columns(2)

    with col1:
        workclass = st.selectbox("Workclass", categories["workclass"])
        education = st.selectbox("Education", categories['education'])
        marital_status = st.selectbox("Marital Status", categories['marital_status'])
        occupation = st.selectbox("Occupation", categories['occupation'])
        relationship = st.selectbox("Relationship", categories['relationship'])
        race = st.selectbox("Race", categories['race'])

    with col2:
        sex = st.selectbox("Sex", categories['sex'])
        age = st.number_input("Age", 18, 100)
        education_num = st.number_input("Education-num", 1, 16)
        capital_gain = st.number_input("Capital gain", 0)
        capital_loss = st.number_input("Capital loss", 0)
        hours = st.number_input("Hours per week", 1, 100)

if st.button("Сделать предсказание"):
    data = {
        "age": age,
        "fnlwgt": 100000,
        "education_num": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours,
        "workclass": workclass,
        "education": education,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex
    }

    try:
        response = requests.post(API_URL + "/predict", json=data)
        res = response.json()["income_greater_than_50k"]

        if res:
            st.success("Доход выше 50K в год")
        else:
            st.error("Доход не превышает 50K в год")

    except Exception as e:
        st.error(f"упс")
