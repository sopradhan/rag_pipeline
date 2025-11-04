import sqlite3
import os

def modify_incident_logs():
    DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Drop existing view if exists
        print("Dropping existing views...")
        cursor.execute("DROP VIEW IF EXISTS active_incidents")
        
        # 1. Rename existing table
        print("Backing up current table...")
        cursor.execute("ALTER TABLE incident_logs RENAME TO incident_logs_old")
        
        # 2. Create new table without embedding_id and severity
        print("Creating new incident_logs table without embedding_id and severity columns...")
        cursor.execute('''
        CREATE TABLE incident_logs (
            id TEXT PRIMARY KEY,
            incident_json JSON NOT NULL,
            source_type TEXT NOT NULL,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK (json_valid(incident_json))
        )
        ''')
        
        # 3. Copy data from old table to new table
        print("Migrating data to new table structure...")
        cursor.execute('''
        INSERT INTO incident_logs (id, incident_json, source_type, status, created_at, updated_at)
        SELECT id, incident_json, source_type, status, created_at, updated_at
        FROM incident_logs_old
        ''')
        
        # 4. Drop old table
        print("Removing old table...")
        cursor.execute("DROP TABLE incident_logs_old")
        
        # 5. Create indices
        print("Creating indices...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incident_logs_status ON incident_logs(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incident_logs_source ON incident_logs(source_type)')
        
        # Commit transaction
        conn.commit()
        print("Successfully modified incident_logs table!")
        
        # Show table info
        cursor.execute("PRAGMA table_info(incident_logs)")
        columns = cursor.fetchall()
        print("\nNew table structure:")
        for col in columns:
            print(f"- {col[1]} ({col[2]})")
        
        # Show record count
        cursor.execute("SELECT COUNT(*) FROM incident_logs")
        count = cursor.fetchone()[0]
        print(f"\nTotal records in incident_logs: {count}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    modify_incident_logs()