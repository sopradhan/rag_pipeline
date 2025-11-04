import sqlite3
import os
import json
from datetime import datetime

def init_technical_database():
    db_path = "D:/incident_management/data/sqlite/incident_management.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute('DROP TABLE IF EXISTS incidents')
    cursor.execute('DROP TABLE IF EXISTS users')
    cursor.execute('DROP TABLE IF EXISTS roles')
    cursor.execute('DROP TABLE IF EXISTS audit_log')
    
    # Create incident_logs table
    cursor.execute('''
    CREATE TABLE incident_logs (
        id TEXT PRIMARY KEY,
        incident_json JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_type TEXT NOT NULL,  -- API, UI, EMAIL, MONITORING_TOOL
        status TEXT DEFAULT 'new',  -- new, processing, classified, resolved
        embedding_id TEXT,          -- Reference to vector store
        CHECK (json_valid(incident_json))
    )
    ''')
    
    # Create system_metrics table
    cursor.execute('''
    CREATE TABLE system_metrics (
        id TEXT PRIMARY KEY,
        metrics_json JSON NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        source_system TEXT NOT NULL,  -- Azure, AWS, GCP, On-Premise
        component_type TEXT NOT NULL,  -- CPU, Memory, Network, Disk, etc.
        alert_threshold JSON,         -- Threshold configuration
        is_alert_triggered BOOLEAN DEFAULT 0,
        CHECK (json_valid(metrics_json)),
        CHECK (json_valid(alert_threshold))
    )
    ''')
    
    # Create alert_emails table
    cursor.execute('''
    CREATE TABLE alert_emails (
        id TEXT PRIMARY KEY,
        email_payload_json JSON NOT NULL,
        received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed_at TIMESTAMP,
        source_email TEXT NOT NULL,
        alert_type TEXT NOT NULL,    -- SECURITY, PERFORMANCE, SYSTEM, etc.
        incident_id TEXT,            -- Reference to related incident
        processing_status TEXT DEFAULT 'new',  -- new, processed, failed
        CHECK (json_valid(email_payload_json))
    )
    ''')

    # Create processing_queue table
    cursor.execute('''
    CREATE TABLE processing_queue (
        id TEXT PRIMARY KEY,
        task_json JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        priority INTEGER DEFAULT 1,
        status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
        retries INTEGER DEFAULT 0,
        last_attempt TIMESTAMP,
        result_json JSON,
        CHECK (json_valid(task_json)),
        CHECK (json_valid(result_json))
    )
    ''')

    # Create vector_embeddings table for reference
    cursor.execute('''
    CREATE TABLE vector_embeddings (
        id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,    -- incident, metric, email
        source_id TEXT NOT NULL,      -- Reference to source record
        embedding_vector BLOB,        -- Store binary vector data
        embedding_model TEXT NOT NULL, -- Model used for embedding
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create indexes for performance
    cursor.execute('CREATE INDEX idx_incident_logs_status ON incident_logs(status)')
    cursor.execute('CREATE INDEX idx_incident_logs_source ON incident_logs(source_type)')
    cursor.execute('CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp)')
    cursor.execute('CREATE INDEX idx_system_metrics_alerts ON system_metrics(is_alert_triggered)')
    cursor.execute('CREATE INDEX idx_alert_emails_status ON alert_emails(processing_status)')
    cursor.execute('CREATE INDEX idx_processing_queue_status ON processing_queue(status)')
    cursor.execute('CREATE INDEX idx_processing_queue_priority ON processing_queue(priority)')
    
    # Sample trigger for updated_at
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS update_incident_timestamp 
    AFTER UPDATE ON incident_logs
    BEGIN
        UPDATE incident_logs SET updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.id;
    END;
    ''')

    # Create views for common queries
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS active_incidents AS
    SELECT 
        id,
        json_extract(incident_json, '$.severity') as severity,
        json_extract(incident_json, '$.service') as service,
        created_at,
        updated_at,
        status
    FROM incident_logs
    WHERE status != 'resolved'
    ''')

    cursor.execute('''
    CREATE VIEW IF NOT EXISTS recent_alerts AS
    SELECT 
        m.id,
        m.source_system,
        m.component_type,
        m.timestamp,
        json_extract(m.metrics_json, '$.value') as metric_value,
        json_extract(m.alert_threshold, '$.threshold') as threshold
    FROM system_metrics m
    WHERE m.is_alert_triggered = 1
    AND m.timestamp >= datetime('now', '-24 hours')
    ''')

    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Technical database initialized successfully at: {db_path}")
    print("\nCreated tables:")
    print("- incident_logs: Store incident data with JSON payload")
    print("- system_metrics: Store system metrics with JSON payload")
    print("- alert_emails: Store alert email data with JSON payload")
    print("- processing_queue: Manage processing tasks")
    print("- vector_embeddings: Store embedding references")
    print("\nCreated views:")
    print("- active_incidents: Shows all non-resolved incidents")
    print("- recent_alerts: Shows alerts from last 24 hours")

if __name__ == "__main__":
    init_technical_database()