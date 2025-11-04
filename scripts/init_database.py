import sqlite3
import os
import json
from datetime import datetime

def init_database():
    db_path = "D:/incident_management/data/sqlite/incident_management.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create incidents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        correlation_id TEXT,
        resource_id TEXT,
        service TEXT,
        region TEXT,
        category TEXT,
        severity TEXT,
        status TEXT,
        detected_time TIMESTAMP,
        resolution_time TIMESTAMP,
        alert_data JSON,
        metadata JSON,
        actions_taken JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create roles table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        permissions JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create audit_log table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        resource_type TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        details JSON,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_correlation_id ON incidents(correlation_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_service ON incidents(service)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)')

    # Insert default admin role
    default_permissions = json.dumps([
        "view_all_incidents",
        "create_incident",
        "update_incident",
        "delete_incident",
        "manage_users",
        "view_audit_log"
    ])
    
    cursor.execute('''
    INSERT OR IGNORE INTO roles (name, permissions) 
    VALUES (?, ?)
    ''', ('admin', default_permissions))

    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database initialized successfully at: {db_path}")
    print("Created tables: incidents, users, roles, audit_log")

if __name__ == "__main__":
    init_database()