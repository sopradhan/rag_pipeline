import sqlite3
import json
from tabulate import tabulate
from datetime import datetime

class DBViewer:
    def __init__(self, db_path="data/sqlite/incident_management.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

    def get_tables(self):
        """Get list of all tables in the database"""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in self.cursor.fetchall()]

    def view_table(self, table_name, limit=10):
        """View contents of a specific table"""
        try:
            # Get column names
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in self.cursor.fetchall()]

            # Get data
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
            rows = self.cursor.fetchall()
            
            # Convert rows to list of dicts
            data = []
            for row in rows:
                row_dict = dict(row)
                # Handle JSON fields
                for key, value in row_dict.items():
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            row_dict[key] = json.dumps(json.loads(value), indent=2)
                        except json.JSONDecodeError:
                            pass
                data.append(row_dict)

            # Print table
            print(f"\n=== Table: {table_name} ===")
            print(tabulate(data, headers="keys", tablefmt="grid"))
            print(f"\nShowing {len(rows)} of", 
                  self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0],
                  "records")

        except sqlite3.Error as e:
            print(f"Error viewing table {table_name}: {e}")

    def execute_query(self, query):
        """Execute a custom SQL query"""
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if rows:
                print("\n=== Query Results ===")
                print(tabulate([dict(row) for row in rows], headers="keys", tablefmt="grid"))
                print(f"\nFound {len(rows)} records")
            else:
                print("No results found")
        except sqlite3.Error as e:
            print(f"Error executing query: {e}")

def main():
    viewer = DBViewer()
    try:
        viewer.connect()
        
        # Get all tables
        tables = viewer.get_tables()
        print("\nAvailable tables:", ", ".join(tables))

        while True:
            print("\nOptions:")
            print("1. View all tables")
            print("2. View specific table")
            print("3. Execute custom query")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                for table in tables:
                    viewer.view_table(table)
            
            elif choice == '2':
                print("\nAvailable tables:", ", ".join(tables))
                table_name = input("Enter table name to view: ")
                limit = input("Enter number of records to view (default 10): ")
                limit = int(limit) if limit.isdigit() else 10
                if table_name in tables:
                    viewer.view_table(table_name, limit)
                else:
                    print("Table not found!")
            
            elif choice == '3':
                query = input("\nEnter your SQL query: ")
                viewer.execute_query(query)
            
            elif choice == '4':
                break
            
            else:
                print("Invalid choice!")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        viewer.close()

if __name__ == "__main__":
    main()