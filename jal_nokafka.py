import time, threading, random, json, queue
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# In-memory queues for IoT data and events
iot_data_queue = queue.Queue()
anomaly_events_queue = queue.Queue()
prediction_events_queue = queue.Queue()
iot_actions_queue = queue.Queue()

# Edge AI: Anomaly Detection

def train_edge_model(seed=42, n=800):
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        milk = np.random.normal(18, 6)
        cow_temp = np.random.normal(38.6, 0.6)
        feed = np.random.normal(24, 7)
        amb_temp = np.random.normal(22, 8)
        anomaly = 1 if milk<10 or milk>32 or cow_temp>39.5 or feed<8 else 0
        rows.append([milk, cow_temp, feed, amb_temp, anomaly])
    df = pd.DataFrame(rows, columns=["milk","cow_temp","feed","amb_temp","anomaly"])
    X = df[["milk","cow_temp","feed","amb_temp"]].values
    y = df["anomaly"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(Xs, y)
    return {'clf': clf, 'scaler': scaler}

edge_model = train_edge_model()

def detect_anomaly(iot_event):
    feats = [iot_event['milk_yield'], iot_event['cow_temp'], iot_event['feed_intake'], iot_event['ambient_temp']]
    X = np.array([feats])
    Xs = edge_model['scaler'].transform(X)
    pred = int(edge_model['clf'].predict(Xs)[0])
    prob = float(edge_model['clf'].predict_proba(Xs)[0][1])
    return {'anomaly': pred, 'prob': prob}

# IoT Simulator

def simulate_iot_data():
    while True:
        evt = {
            "farm_id": f"farm_{random.randint(1,5)}",
            "milk_yield": round(random.gauss(18,8),2),
            "cow_temp": round(random.gauss(38.6,1),2),
            "feed_intake": round(random.gauss(24,7),2),
            "ambient_temp": round(random.gauss(22,9),2),
            "timestamp": time.time()
        }
        iot_data_queue.put(evt)
        time.sleep(0.5)

# IoT Consumer & Processing

def process_iot_stream():
    while True:
        if not iot_data_queue.empty():
            iot_event = iot_data_queue.get()
            anomaly = detect_anomaly(iot_event)
            evt_out = {**iot_event, **anomaly}
            anomaly_events_queue.put(evt_out)
            # LSTM store history (stub)
            # ...existing code...
            # If anomaly, call LLM for recovery steps (stub)
            if anomaly['anomaly']==1:
                recovery = {"steps": ["Check local water valve","Dispatch tanker to farm","Alert repair crew"], "actuator_commands": []}
                iot_actions_queue.put({"farm_id":iot_event['farm_id'],"recovery":recovery})
        time.sleep(0.1)

# Streamlit Dashboard

def run_dashboard():
    st.title("JalMitraAI Real-time Dashboard (No Kafka)")
    st.markdown("## IoT Sensor & Edge AI Anomalies")
    while True:
        iot_vals = list(iot_data_queue.queue)[-5:]
        anomaly_vals = list(anomaly_events_queue.queue)[-5:]
        forecast_vals = [] # stub
        llm_vals = list(iot_actions_queue.queue)[-5:]
        st.subheader("IoT Events")
        st.json(iot_vals)
        st.subheader("Anomalies Detected (Edge AI)")
        st.json(anomaly_vals)
        st.subheader("Forecast (LSTM)")
        st.json(forecast_vals)
        st.subheader("LLM Recovery & Actuator Actions")
        st.json(llm_vals)
        time.sleep(2)

if __name__=="__main__":
    threading.Thread(target=simulate_iot_data, daemon=True).start()
    threading.Thread(target=process_iot_stream, daemon=True).start()
    run_dashboard()
