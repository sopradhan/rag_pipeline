import sqlite3
import json
from tabulate import tabulate

def show_sample_logs():
    conn = sqlite3.connect("D:/incident_management/data/sqlite/incident_management.db")
    cursor = conn.cursor()
    
    # Get one sample of each type
    for source_type in ['ActivityLog', 'MetricLog', 'DiagnosticLog']:
        print(f"\n=== Sample {source_type} ===")
        cursor.execute("""
            SELECT incident_json 
            FROM incident_logs 
            WHERE source_type = ? 
            LIMIT 1
        """, (source_type,))
        
        row = cursor.fetchone()
        if row:
            log = json.loads(row[0])
            if 'tags' in log:
                print("\nResource Tags:")
                print(json.dumps(log['tags'], indent=2))
            print("\nLog Content:")
            print(json.dumps(log, indent=2))
    
    # Show environment distribution
    print("\n=== Environment Distribution ===")
    cursor.execute("""
        SELECT 
            json_extract(incident_json, '$.tags.Environment') as env,
            COUNT(*) as count
        FROM incident_logs
        GROUP BY json_extract(incident_json, '$.tags.Environment')
    """)
    
    env_dist = cursor.fetchall()
    if env_dist:
        print("\nLogs by Environment:")
        for env, count in env_dist:
            print(f"{env}: {count} logs")
    
    conn.close()

if __name__ == "__main__":
    show_sample_logs()