import sqlite3

# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

def create_severity_rules_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create severity rules table
    cursor.execute("DROP TABLE IF EXISTS severity_rules")
    cursor.execute("""
    CREATE TABLE severity_rules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT NOT NULL,
        severity_level TEXT NOT NULL,
        base_score REAL NOT NULL,
        category TEXT,
        description TEXT,
        environment TEXT,
        UNIQUE(pattern, severity_level, environment)
    )
    """)
    
    # Comprehensive rules data with environment consideration
    rules_data = [
        # S1 (Critical) - Production
        ("complete system failure", "S1", 100, "availability", "Complete system unavailability", "prod"),
        ("data corruption detected", "S1", 95, "data", "Critical data integrity issue", "prod"),
        ("security breach detected", "S1", 100, "security", "Security compromise", "prod"),
        ("production database failure", "S1", 95, "database", "Critical database issue", "prod"),
        ("ssl certificate expired", "S1", 90, "security", "SSL security issue", "prod"),
        ("memory exhaustion", "S1", 95, "resource", "Critical resource issue", "prod"),
        ("authentication system down", "S1", 95, "security", "Authentication failure", "prod"),
        ("critical api gateway failure", "S1", 90, "network", "API connectivity issue", "prod"),
        
        # S1 (Critical) - UAT
        ("test environment down", "S1", 85, "availability", "Test environment unavailable", "uat"),
        ("integration test failure", "S1", 85, "testing", "Critical test failure", "uat"),
        ("uat database corruption", "S1", 85, "database", "Test database issue", "uat"),
        
        # S1 (Critical) - Dev
        ("development environment crash", "S1", 80, "availability", "Dev environment issue", "dev"),
        ("source control system down", "S1", 80, "tools", "Critical dev tool issue", "dev"),
        
        # S2 (High) - Production
        ("partial system outage", "S2", 75, "availability", "Partial availability issue", "prod"),
        ("high latency detected", "S2", 75, "performance", "Performance degradation", "prod"),
        ("database performance degradation", "S2", 75, "database", "DB performance issue", "prod"),
        ("elevated error rate", "S2", 70, "reliability", "Increased errors", "prod"),
        ("network connectivity issues", "S2", 70, "network", "Network problems", "prod"),
        ("memory leak detected", "S2", 70, "resource", "Resource leak", "prod"),
        
        # S2 (High) - UAT
        ("test pipeline failure", "S2", 65, "testing", "Test pipeline issue", "uat"),
        ("integration error", "S2", 65, "testing", "Integration issue", "uat"),
        ("performance test failure", "S2", 65, "performance", "Performance test issue", "uat"),
        
        # S2 (High) - Dev
        ("build pipeline failure", "S2", 60, "build", "Build system issue", "dev"),
        ("development database issue", "S2", 60, "database", "Dev DB problem", "dev"),
        
        # S3 (Medium) - Production
        ("minor service degradation", "S3", 55, "performance", "Minor performance issue", "prod"),
        ("increased response time", "S3", 50, "performance", "Slower responses", "prod"),
        ("warning threshold reached", "S3", 50, "monitoring", "Warning alert", "prod"),
        ("resource utilization high", "S3", 50, "resource", "Resource pressure", "prod"),
        
        # S3 (Medium) - UAT/Dev
        ("test warning", "S3", 45, "testing", "Test environment warning", "uat"),
        ("development warning", "S3", 40, "development", "Development warning", "dev"),
        
        # S4 (Low) - All Environments
        ("informational alert", "S4", 20, "info", "Information only", "prod"),
        ("scheduled maintenance", "S4", 15, "maintenance", "Planned maintenance", "prod"),
        ("minor configuration change", "S4", 20, "config", "Config update", "prod"),
        ("routine operation", "S4", 10, "operations", "Regular activity", "prod"),
        ("test information", "S4", 10, "testing", "Test info", "uat"),
        ("development notification", "S4", 10, "development", "Dev notification", "dev")
    ]
    
    # Insert rules
    cursor.executemany("""
    INSERT INTO severity_rules 
    (pattern, severity_level, base_score, category, description, environment)
    VALUES (?, ?, ?, ?, ?, ?)
    """, rules_data)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_severity_rules_table()