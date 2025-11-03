import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from langchain_community.llms import HuggingFaceHub   # ✅ use HuggingFaceHub LLM
import httpx
import altair as alt
import time

# =============================
# Setup LLM (Hugging Face)
# =============================
import os
from dotenv import load_dotenv

@st.cache_resource
def setup_llm():
    # Load environment variables
    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_api_key:
        raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

    # You can replace the model below with any other HF model that supports text generation
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=hf_api_key,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 200}
    )
    return llm

# =============================
# Parameters
# =============================
np.random.seed(42)
plants = ['Plant_1', 'Plant_2', 'Plant_3']
substations = ['Substation_1', 'Substation_2', 'Substation_3']
features = ['Voltage (kV)', 'Current (A)', 'Temperature (°C)', 'Humidity (%)']

# =============================
# Fault logic functions
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
        return 'Temperature (°C)', row['Temperature (°C)']
    elif fault_label == 'Undervoltage':
        return 'Voltage (kV)', row['Voltage (kV)']
    elif fault_label in ['Power Surge', 'Low Power']:
        return 'Current (A)', row['Current (A)']
    else:
        return 'None', 'N/A'

# =============================
# Synthetic Data Generation
# =============================
def generate_training_data(n=500, agent_type="Plant"):
    data = []
    for _ in range(n):
        if agent_type == "Plant":
            entity = random.choice(plants)
            entity_col = "Plant_ID"
        else:
            entity = random.choice(substations)
            entity_col = "Substation_ID"

        voltage = np.random.normal(11.5, 0.5)
        current = np.random.normal(115, 10)
        temperature = np.random.normal(40, 12)
        humidity = np.random.normal(45, 10)

        df_row = pd.DataFrame([[entity, voltage, current, temperature, humidity]],
                              columns=[entity_col] + features)
        df_row['Fault Label'] = df_row.apply(determine_fault, axis=1)
        data.append(df_row)
    return pd.concat(data, ignore_index=True)

# =============================
# Model Training
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
    reason_feature, reason_value = explain_fault(df_row.iloc[0], fault_label)
    return fault_label, reason_feature, reason_value

# =============================
# Cached Models
# =============================
@st.cache_data
def train_power_plant_model():
    df_train = generate_training_data(1000, agent_type="Plant")
    return train_model(df_train)

@st.cache_data
def train_substation_model():
    df_train = generate_training_data(1000, agent_type="Substation")
    return train_model(df_train)

# =============================
# Real-time Simulation & Monitoring
# =============================
def real_time_monitoring(model_dict, entity_type="Plant", llm=None, num_iterations=50, interval=1):
    st.subheader("⚡ Real-Time Fault Monitoring")

    raw_placeholder = st.empty()
    table_placeholder = st.empty()
    chart_placeholder = st.empty()

    raw_data = pd.DataFrame(columns=['Time', 'Entity', 'Voltage (kV)', 'Current (A)', 'Temperature (°C)', 'Humidity (%)'])
    pred_data = pd.DataFrame(columns=['Time', 'Entity', 'Fault', 'Reason Feature', 'Reason Value', 'Corrective Measure'])
    trend_data = pd.DataFrame(columns=['Time', 'Voltage (kV)', 'Current (A)', 'Temperature (°C)', 'Fault'])

    for i in range(num_iterations):
        timestamp = pd.Timestamp.now()

        # Bias: 90% faulty, 10% normal
        if random.random() < 0.9:
            voltage = np.random.normal(10, 1)
            current = np.random.normal(160, 20)
            temperature = np.random.normal(65, 10)
        else:
            voltage = np.random.normal(11.5, 0.5)
            current = np.random.normal(115, 10)
            temperature = np.random.normal(40, 5)

        humidity = np.random.normal(45, 10)
        entity_id = random.choice(plants if entity_type == "Plant" else substations)

        raw_row = pd.DataFrame([{
            'Time': timestamp,
            'Entity': entity_id,
            'Voltage (kV)': voltage,
            'Current (A)': current,
            'Temperature (°C)': temperature,
            'Humidity (%)': humidity
        }])
        raw_data = pd.concat([raw_data, raw_row], ignore_index=True)

        row_for_model = pd.DataFrame([{
            'Voltage (kV)': voltage,
            'Current (A)': current,
            'Temperature (°C)': temperature,
            'Humidity (%)': humidity
        }])
        fault_label, reason_feature, reason_value = predict_fault(row_for_model, model_dict)

        # AI reasoning
        if llm:
            prompt = f"""You are an electrical fault diagnosis expert.
Analyze the following reading and explain the root cause of the fault in simple terms.

{entity_type}: {entity_id}
Fault Detected: {fault_label}
Key Parameter: {reason_feature} = {reason_value}
"""
            ai_explanation = llm(prompt)
        else:
            ai_explanation = "LLM not available"

        pred_row = pd.DataFrame([{
            'Time': timestamp,
            'Entity': entity_id,
            'Fault': fault_label,
            'Reason Feature': reason_feature,
            'Reason Value': reason_value,
            'Corrective Measure': ai_explanation
        }])
        pred_data = pd.concat([pred_data, pred_row], ignore_index=True)

        trend_row = pd.DataFrame([{
            'Time': timestamp,
            'Voltage (kV)': voltage,
            'Current (A)': current,
            'Temperature (°C)': temperature,
            'Fault': fault_label
        }])
        trend_data = pd.concat([trend_data, trend_row], ignore_index=True)

        raw_placeholder.dataframe(raw_data.tail(10))
        table_placeholder.dataframe(pred_data.tail(10))

        chart = (
            alt.Chart(trend_data)
            .transform_fold(['Voltage (kV)', 'Current (A)', 'Temperature (°C)'], as_=['Parameter', 'Value'])
            .mark_line()
            .encode(
                x='Time:T',
                y='Value:Q',
                color='Parameter:N',
                tooltip=['Time:T', 'Parameter:N', 'Value:Q', 'Fault:N']
            )
            .properties(width=800, height=400)
        )
        chart_placeholder.altair_chart(chart, use_container_width=True)

        time.sleep(interval)

# =============================
# Streamlit App
# =============================
def main():
    st.set_page_config(page_title="⚡Grid Guardian", page_icon="⚡", layout="wide")
    st.title("⚡ Grid Guardian")
    st.markdown("Upload CSV or simulate **real-time sensor data** for Power Plant or Substation with live AI reasoning.")

    agent_type = st.radio("Select Agent:", ["Power Plant", "Distribution Substation"], horizontal=True)

    with st.spinner("Training model..."):
        if agent_type == "Power Plant":
            model_dict = train_power_plant_model()
            entity_name = "Plant"
        else:
            model_dict = train_substation_model()
            entity_name = "Substation"

    try:
        llm = setup_llm()
        st.success("Hugging Face LLM connected")
    except Exception as e:
        st.warning(f"LLM not available: {str(e)}")
        llm = None

    st.subheader("🔄 Data Modes")
    option = st.radio("Choose how to feed test data:", ["Simulate Real-Time Data", "Upload CSV"])

    if option == "Upload CSV":
        uploaded_file = st.file_uploader(f"Upload CSV with {entity_name} test data", type="csv")
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.subheader("📂 Uploaded Data Preview")
            st.dataframe(df_input.head())
    else:
        real_time_monitoring(model_dict, entity_type=entity_name, llm=llm, num_iterations=50, interval=1)

if __name__ == "__main__":
    main()
