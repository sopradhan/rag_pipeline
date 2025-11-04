import sqlite3
import os

def verify_sqlite():
    print("Checking SQLite version and database...")
    
    # Create the database directory if it doesn't exist
    db_dir = "data/sqlite"
    os.makedirs(db_dir, exist_ok=True)
    
    # Database path
    db_path = "D:/incident_management/data/sqlite/incident_management.db"
    
    try:
        # Connect to database (will create if doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get SQLite version
        cursor.execute('SELECT SQLITE_VERSION()')
        version = cursor.fetchone()
        print(f"SQLite Version: {version[0]}")
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            print("\nExisting tables:")
            for table in tables:
                print(f"- {table[0]}")
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"  Records: {count}")
        else:
            print("\nNo tables found in the database.")
            
        conn.close()
        print("\nSQLite verification completed successfully!")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_sqlite()