import sqlite3
import os

def update_schema():
    DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Backup current data
    print("Creating backup of current incident_logs...")
    cursor.execute("ALTER TABLE incident_logs RENAME TO incident_logs_backup")
    
    # Create new table with severity column
    print("Creating updated incident_logs table...")
    cursor.execute('''
    CREATE TABLE incident_logs (
        id TEXT PRIMARY KEY,
        incident_json JSON NOT NULL,
        source_type TEXT NOT NULL,
        status TEXT DEFAULT 'new',
        severity TEXT NOT NULL,
        embedding_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CHECK (json_valid(incident_json))
    )
    ''')
    
    # Copy data from backup with default severity
    print("Restoring data with default severity...")
    cursor.execute('''
    INSERT INTO incident_logs (id, incident_json, source_type, status, embedding_id, created_at, updated_at, severity)
    SELECT id, incident_json, source_type, status, embedding_id, created_at, updated_at, 'MEDIUM'
    FROM incident_logs_backup
    ''')
    
    # Drop backup table
    cursor.execute("DROP TABLE incident_logs_backup")
    
    # Create index on severity
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_incident_severity ON incident_logs(severity)')
    
    conn.commit()
    conn.close()
    print("Schema update completed successfully!")

if __name__ == '__main__':
    update_schema()