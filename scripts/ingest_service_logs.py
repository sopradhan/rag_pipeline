import sqlite3
import json
import uuid
import random
from datetime import datetime, timedelta

DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

# Helper functions
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def gen_uuid():
    return str(uuid.uuid4())

# Sample generators for different log types
def sample_activity_log():
    # ARM operations, role changes, provisioning
    return {
        "eventTimestamp": now_iso(),
        "operationName": random.choice(["Microsoft.Resources/subscriptions/resourceGroups/write", "Microsoft.Authorization/roleAssignments/write", "Microsoft.Resources/deployments/write"]),
        "status": random.choice(["Success", "Failed"]),
        "caller": random.choice(["user@contoso.com", "svc-app@contoso.com"]),
        "resourceId": f"/subscriptions/{gen_uuid()}/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-{random.randint(1,999)}",
        "details": {"changes": ["created","updated"]}
    }

def sample_security_log():
    return {
        "time": now_iso(),
        "alertType": random.choice(["SuspiciousLogin", "MalwareDetected", "CryptoFailure"]),
        "severity": random.choice(["High","Medium","Low"]),
        "resource": "Microsoft.KeyVault/vaults/kv-01",
        "description": "Detected suspicious activity from unknown IP",
        "threatDetails": {"ip": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"}
    }

def sample_service_health():
    return {
        "timestamp": now_iso(),
        "service": random.choice(["Azure SQL", "Key Vault", "App Service"]),
        "event": random.choice(["PlannedMaintenance", "IncidentReported", "AssistedRecovery"]),
        "impact": random.choice(["None","Partial","Full"]),
        "message": "Planned maintenance scheduled"
    }

def sample_apim_gateway_log():
    return {
        "time": now_iso(),
        "requestId": gen_uuid(),
        "method": random.choice(["GET","POST","PUT"]),
        "path": "/api/v1/orders",
        "status": random.choice([200,400,401,429,500]),
        "latencyMs": random.randint(1,2000),
        "backend": "orders-svc"
    }

def sample_sql_audit_log():
    return {
        "time": now_iso(),
        "server": "sqldb-prod-01",
        "user": random.choice(["dbuser","appsvc"]),
        "action": random.choice(["SELECT","INSERT","DELETE","ALTER"]),
        "success": random.choice([True, False]),
        "query": "SELECT * FROM orders WHERE id = 123"
    }

def sample_cosmos_log():
    return {
        "time": now_iso(),
        "operation": random.choice(["Read","Write","Query"]),
        "throughputRUs": random.randint(1,10000),
        "container": "orders",
        "statusCode": random.choice([200,429,500])
    }

def sample_microservice_log():
    # Structured JSON logs from app containers
    return {
        "ts": now_iso(),
        "service": "orders-svc",
        "level": random.choice(["INFO","WARN","ERROR"]),
        "message": "Order processing failed",
        "traceId": gen_uuid(),
        "details": {"orderId": random.randint(1000,9999), "attempt": 1}
    }

def sample_trace_span():
    return {
        "traceId": gen_uuid(),
        "spanId": gen_uuid(),
        "parentId": None,
        "name": random.choice(["HTTP GET /api/orders","DB SELECT orders"]),
        "durationMs": random.randint(1,2000),
        "status": random.choice(["OK","ERROR"]),
        "timestamp": now_iso()
    }

def sample_databricks_audit():
    return {
        "time": now_iso(),
        "workspace": "databricks-prod",
        "eventType": random.choice(["JobStart","JobEnd","ClusterResize"]),
        "user": random.choice(["analyst@contoso.com","svc-job@contoso.com"]),
        "details": {"jobId": random.randint(100,999)}
    }

def sample_diagnostic_job_failure():
    return {
        "time": now_iso(),
        "pipeline": "etl-orders",
        "runId": gen_uuid(),
        "status": "Failed",
        "error": "Timeout while connecting to source"
    }

def sample_vm_metric():
    return {
        "time": now_iso(),
        "vm": f"vm-{random.randint(1,50)}",
        "cpuPercent": round(random.uniform(0,100),2),
        "memoryMB": random.randint(512,65536),
        "diskPercent": round(random.uniform(0,100),2),
        "networkInKB": random.randint(0,100000)
    }

def sample_container_metric():
    return {
        "time": now_iso(),
        "pod": f"orders-{random.randint(1,20)}",
        "container": "orders",
        "cpuCores": round(random.uniform(0,4),2),
        "memoryMB": random.randint(50,2048),
        "restarts": random.randint(0,5)
    }

# Mapping to insert into DB
INSERT_COUNT_PER_TYPE = 5

def insert_into_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # For each type, generate and insert sample records
    # 1. Activity / Admin / Security / Service health -> incident_logs
    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_activity_log()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Activity", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_security_log()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Security", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_service_health()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "ServiceHealth", "new"))

    # 2. API Management / SQL Audit / Cosmos / Microservice / Databricks / Diagnostic -> alert_emails or incident_logs depending
    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_apim_gateway_log()
        # These are structured gateway logs; treat as alerts if status >=500 or throttling
        email_payload = {
            "subject": "APIM Gateway Alert",
            "body": rec,
            "detected": now_iso()
        }
        cursor.execute("INSERT INTO alert_emails (id, email_payload_json, source_email, alert_type) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(email_payload), "apim@contoso.com", "API_MANAGEMENT"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_sql_audit_log()
        # SQL audit logs are incidents
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "SQLAudit", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_cosmos_log()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "CosmosDB", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_microservice_log()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Microservice", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_trace_span()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Tracing", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_databricks_audit()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Databricks", "new"))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_diagnostic_job_failure()
        cursor.execute("INSERT INTO incident_logs (id, incident_json, source_type, status) VALUES (?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), "Diagnostic", "new"))

    # 3. System metrics -> system_metrics
    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_vm_metric()
        cursor.execute("INSERT INTO system_metrics (id, metrics_json, timestamp, source_system, component_type, alert_threshold, is_alert_triggered) VALUES (?,?,?,?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), rec['time'], 'Azure Monitor', 'VM', json.dumps({"cpu":85}), 1 if rec['cpuPercent']>85 else 0))

    for _ in range(INSERT_COUNT_PER_TYPE):
        rec = sample_container_metric()
        cursor.execute("INSERT INTO system_metrics (id, metrics_json, timestamp, source_system, component_type, alert_threshold, is_alert_triggered) VALUES (?,?,?,?,?,?,?)",
                       (gen_uuid(), json.dumps(rec), rec['time'], 'Kubernetes', 'Container', json.dumps({"cpuCores":3}), 1 if rec['cpuCores']>3 else 0))

    conn.commit()
    conn.close()
    print("Inserted sample logs into DB:")
    print(f"- Incident logs: {INSERT_COUNT_PER_TYPE * 9} (multiple categories)")
    print(f"- Alert emails: {INSERT_COUNT_PER_TYPE}")
    print(f"- System metrics: {INSERT_COUNT_PER_TYPE * 2}")

if __name__ == '__main__':
    insert_into_db()