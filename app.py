import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf

# ---------- Load saved objects ----------
rf = joblib.load("rf_regressor.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

nn_model = tf.keras.models.load_model("nn_model.h5")

with open("feature_columns.json") as f:
    feature_columns = json.load(f)

# ---------- UI ----------
st.title("🔋 EV Battery Degradation Predictor")

st.header("Enter Battery Parameters")

charge_cycles = st.number_input("Charge Cycles", 0, 2000, 400)
avg_temp = st.number_input("Avg Temperature (°C)", 0.0, 100.0, 30.0)
battery_age = st.number_input("Battery Age (months)", 0, 120, 24)
discharge_depth = st.number_input("Discharge Depth", 0, 100, 70)
voltage_var = st.number_input("Voltage Variation", 0.0, 1.0, 0.15)

# ---------- Predict ----------
if st.button("Predict Battery Health"):

    input_dict = {
        "charge_cycles": charge_cycles,
        "avg_temperature_c": avg_temp,
        "battery_age_months": battery_age,
        "discharge_depth": discharge_depth,
        "voltage_variation": voltage_var
    }

    df = pd.DataFrame([input_dict])

    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df_enc)

    # regression
    cap_pred = rf.predict(scaled)[0]

    # classification
    cls = nn_model.predict(scaled).argmax()
    health = le.inverse_transform([cls])[0]

    st.success(f"Remaining Capacity: {cap_pred:.2f}%")
    st.success(f"Battery Health: {health}")
