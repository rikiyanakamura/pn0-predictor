
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Lymph Node Metastasis Predictor", layout="centered")
st.title("Prediction Tool for Lymph Node Metastasis (pN0)")
st.markdown("Enter patient information to predict probability of node-negative disease.")

# 入力項目
age = st.number_input("Age", min_value=18, max_value=100, value=50)
sex = st.selectbox("Sex", ["女性", "男性"])
height = st.number_input("Height (cm)", min_value=130.0, max_value=200.0, value=160.0)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=55.0)
menopause = st.selectbox("Menopause", ["閉経前", "閉経後"])
diagnostic = st.selectbox("Axillary Diagnosis", ["image diagnostic.", "CNB negative", "FNA negative"])
cT = st.selectbox("Clinical T Stage", ["Tis", "T1", "T2", "T3", "T4"])
histology = st.selectbox("CNB Histopathology", ["invasive ductal ca.", "DCIS", "tubular ca.", "mucinus ca.", "invasive Lobular ca.", "microinvasive ca."])
cHG = st.selectbox("cHG", ["1", "2", "3"])
cER = st.slider("ER(%)", 0, 100, 80)
cPgR = st.slider("PgR(%)", 0, 100, 20)
cHER2 = st.selectbox("HER2 Status", ["0", "1+", "2+", "3+"])
her2_expr = st.selectbox("HER2 protein", ["negative", "positive"])
us_size = st.number_input("Ultrasound Tumor Diameter (mm)", min_value=0.0, max_value=100.0, value=15.0)

# 入力をデータフレーム化
input_dict = {
    "age": age,
    "Sex": sex,
    "high(cm)": height,
    "Weight(kg)": weight,
    "menopause": menopause,
    "Diagnositc": diagnostic,
    "cT": cT,
    "CNB Histopathology": histology,
    "cHG": cHG,
    "cER": cER,
    "cPgR": cPgR,
    "cHER2": cHER2,
    "HER2 expression": her2_expr,
    "US　size(mm)": us_size
}
input_df = pd.DataFrame([input_dict])

# モデル読み込みと予測
model = joblib.load("rf_model.pkl")
prediction = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
st.success(f"Predicted probability of lymph node metastasis: {prediction:.2%}")
