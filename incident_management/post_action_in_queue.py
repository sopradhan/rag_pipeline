import sqlite3
import json
import time
import sys
import os
from pathlib import Path

print (Path(__file__))
# Add models directory (parent of database folder) to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
print (BASE_DIR)

from task_queue.jobs_queue import SQLiteQueue
from task_queue.producer import Producer

DB_PATH = r"C:\Users\GENAIKOLGPUSR15\Desktop\Incident_management\incident_db\data\incident_iq.db"

queue = SQLiteQueue(DB_PATH)
    

def fetch_classifier_outputs_json():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT payload_id, environment, severity_id, matched_pattern, 
               source_type, payload, processed_at, corrective_action
        FROM classifier_outputs
        LIMIT 10
    """)
    rows = cursor.fetchall()
    conn.close()

    cols = ["payload_id", "environment", "severity_id", "matched_pattern",
            "source_type", "payload", "processed_at", "corrective_action"]

    data = [dict(zip(cols, row)) for row in rows]
    json_data = json.dumps(data, indent=2, default=str)
    print(json_data)

    task_list = json.loads(json_data)

    producer1 = Producer(queue, "Producer-1", task_list)
    print("Starting producer...\n")
    producer1.start()
    producer1.join()  # Wait for producer to complete

    print("Producer finished.\n")

    return json_data

if __name__ == "__main__":
    fetch_classifier_outputs_json()
