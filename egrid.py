import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import altair as alt
$env:HF_HOME="D:/projectexplored"
$env:TRANSFORMERS_CACHE="D:/projectexplored"

# =============================
# LLM Setup with Hugging Face API Key
# =============================
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

@st.cache_resource
def setup_llm():
    try:
        load_dotenv()
        hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_api_key:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")
        llm = HuggingFaceEndpoint.from_model_id(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            huggingfacehub_api_token=hf_api_key,
            model_kwargs={"temperature":0, "max_new_tokens":200}
        )
        return llm
    except Exception as e:
        st.warning(f"LLM not available: {e}")
        return None

# =============================
# Parameters
# =============================
np.random.seed(42)
plants = ['Plant_1', 'Plant_2', 'Plant_3']
substations = ['Substation_1', 'Substation_2', 'Substation_3']
features = ['Voltage (kV)', 'Current (A)', 'Temperature (°C)', 'Humidity (%)']
baseline = {'Voltage (kV)':11, 'Current (A)':120, 'Temperature (°C)':45, 'Humidity (%)':50}

# =============================
# Fault logic
# =============================
def determine_fault(row):
    if row['Temperature (°C)'] > 60:
        return 'Overheat'
    elif row['Voltage (kV)'] < 10.5:
        return 'Undervoltage'
    elif row['Current (A)'] > 150:
        return 'Power Surge'
    elif row['Current (A)'] < 80:
        return 'Low Power'
    else:
        return 'Normal'

def explain_fault(row, fault_label):
    if fault_label == 'Overheat':
        return 'Temperature (°C)', row['Temperature (°C)'], "Increase cooling, check fan/AC"
    elif fault_label == 'Undervoltage':
        return 'Voltage (kV)', row['Voltage (kV)'], "Check supply line & transformer"
    elif fault_label == 'Power Surge':
        return 'Current (A)', row['Current (A)'], "Inspect breakers & load"
    elif fault_label == 'Low Power':
        return 'Current (A)', row['Current (A)'], "Check load distribution"
    else:
        return 'None', 'N/A', "No action needed"

# =============================
# Data generation
# =============================
def generate_data(n=100, agent_type="Plant"):
    data = []
    for _ in range(n):
        entity = random.choice(plants if agent_type=="Plant" else substations)
        entity_col = "Plant_ID" if agent_type=="Plant" else "Substation_ID"
        voltage = np.random.normal(11.5, 0.5)
        current = np.random.normal(115, 10)
        temperature = np.random.normal(40, 12)
        humidity = np.random.normal(45, 10)

        row = pd.DataFrame([[entity, voltage, current, temperature, humidity]],
                           columns=[entity_col]+features)
        row['Fault Label'] = row.apply(determine_fault, axis=1)
        data.append(row)
    return pd.concat(data, ignore_index=True)

# =============================
# Model training
# =============================
def train_model(df):
    le = LabelEncoder()
    df['Fault_Label_Encoded'] = le.fit_transform(df['Fault Label'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, df['Fault_Label_Encoded'])
    return {'model': rf, 'scaler': scaler, 'le': le}

def predict_fault(df_row, model_dict):
    X_scaled = model_dict['scaler'].transform(df_row[features])
    pred = model_dict['model'].predict(X_scaled)[0]
    fault_label = model_dict['le'].inverse_transform([pred])[0]
    reason_feature, reason_value, corrective = explain_fault(df_row.iloc[0], fault_label)
    return fault_label, reason_feature, reason_value, corrective

# =============================
# Cached models
# =============================
@st.cache_data
def get_models():
    plant_model = train_model(generate_data(100,"Plant"))
    substation_model = train_model(generate_data(100,"Substation"))
    return plant_model, substation_model

# =============================
# Real-time simulation
# =============================
def simulate_real_time(model_dict, entity_type="Plant", llm=None, n=100):
    st.subheader(f"⚡ Real-Time {entity_type} Monitoring (100 records)")
    raw_data = pd.DataFrame(columns=['Time','Entity']+features+['Fault','Corrective Measure'])
    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()

    for i in range(n):
        timestamp = pd.Timestamp.now()
        entity_id = random.choice(plants if entity_type=="Plant" else substations)
        voltage = np.random.normal(11.5,0.5)
        current = np.random.normal(115,10)
        temperature = np.random.normal(40,12)
        humidity = np.random.normal(45,10)

        row = pd.DataFrame([{
            'Time': timestamp,
            'Entity': entity_id,
            'Voltage (kV)': voltage,
            'Current (A)': current,
            'Temperature (°C)': temperature,
            'Humidity (%)': humidity
        }])

        fault_label, feature, value, corrective = predict_fault(row, model_dict)

        # LLM integration with context constraint
        if llm:
            prompt = f"""
You are an expert in power grid fault diagnosis.
Do not provide information outside the given context.
Analyze this reading and provide step-by-step corrective measures.

Fault Detected: {fault_label}
Key Parameter: {feature} = {value}

Return steps concisely.
"""
            try:
                corrective = llm(prompt)
            except Exception:
                corrective = corrective  # fallback

        row['Fault'] = fault_label
        row['Corrective Measure'] = corrective
        raw_data = pd.concat([raw_data,row],ignore_index=True)

        # KPI Metrics
        total_faults = len(raw_data[raw_data['Fault']!='Normal'])
        overheat_count = len(raw_data[raw_data['Fault']=='Overheat'])
        undervoltage_count = len(raw_data[raw_data['Fault']=='Undervoltage'])
        power_surge_count = len(raw_data[raw_data['Fault']=='Power Surge'])
        low_power_count = len(raw_data[raw_data['Fault']=='Low Power'])

        kpi_placeholder.metric("Total Faults", total_faults)
        st.write(f"Overheat: {overheat_count}, Undervoltage: {undervoltage_count}, Power Surge: {power_surge_count}, Low Power: {low_power_count}")

        # Real-time interactive chart with spike alerts
        df_chart = raw_data.copy()
        for feature_name in features:
            df_chart[f"{feature_name}_Status"] = np.where(
                (df_chart[feature_name] > baseline[feature_name]*1.1) | (df_chart[feature_name] < baseline[feature_name]*0.9),
                "Red","Green"
            )

        chart = (
            alt.Chart(df_chart)
            .transform_fold(features, as_=['Parameter','Value'])
            .mark_line()
            .encode(
                x='Time:T',
                y='Value:Q',
                color='Parameter:N',
                tooltip=['Time:T','Parameter:N','Value:Q','Fault:N']
            )
            .properties(width=900,height=400)
        )
        chart_placeholder.altair_chart(chart,use_container_width=True)

        time.sleep(0.1)  # fast demo

    st.subheader("Latest Records")
    st.dataframe(raw_data.tail(10))

# =============================
# Streamlit app
# =============================
def main():
    st.set_page_config(page_title="⚡Indian Grid Dashboard",layout="wide")
    st.title("⚡ Indian Power Grid Dashboard")
    st.markdown("Choose **Stakeholder Dashboard** or **Operator Dashboard**")

    llm = setup_llm()
    plant_model, substation_model = get_models()

    dashboard_type = st.radio("Select Dashboard", ["Stakeholder Dashboard", "Operator Dashboard"], horizontal=True)
    agent_type = st.radio("Select Agent", ["Power Plant","Distribution Substation"], horizontal=True)
    model_dict = plant_model if agent_type=="Power Plant" else substation_model

    simulate_real_time(model_dict, entity_type=agent_type, llm=llm, n=100)

if __name__=="__main__":
    main()
