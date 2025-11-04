import sqlite3
import json
import uuid
import random
from datetime import datetime, timedelta
import ipaddress
import string

DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

class EnhancedLogGenerator:
    def __init__(self):
        self.severity_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        self.severity_weights = [0.1, 0.2, 0.3, 0.4]  # Distribution of severity
        
        # Expanded service components
        self.services = {
            "KeyVault": ["secrets", "keys", "certificates"],
            "Storage": ["blob", "queue", "table", "file"],
            "SQL": ["database", "elastic-pool", "managed-instance"],
            "AppService": ["webapp", "function", "api"],
            "CosmosDB": ["mongodb", "sql", "cassandra", "gremlin"],
            "AKS": ["node-pool", "pod", "service", "ingress"],
            "Network": ["vnet", "gateway", "firewall", "load-balancer"]
        }
        
        # Real-world incident patterns
        self.incident_patterns = {
            "Security": {
                "CRITICAL": [
                    {
                        "title": "Unauthorized Key Vault Access Attempt",
                        "template": "Multiple failed authentication attempts detected from IP {ip}. {attempt_count} failed attempts using invalid credentials for Key Vault {resource}. Source location: {location}. Potential brute force attack detected.",
                        "impact": "Potential unauthorized access to sensitive credentials and certificates.",
                        "recommended_action": "Block source IP, audit Key Vault access logs, rotate potentially compromised credentials."
                    },
                    {
                        "title": "Ransomware Activity Detected",
                        "template": "Suspicious file encryption activity detected on {resource}. High volume of file modifications with entropy changes. Affected paths: {paths}. Potential ransomware strain: {malware_family}.",
                        "impact": "Critical data encryption and business operation disruption risk.",
                        "recommended_action": "Isolate affected systems, initiate incident response plan, restore from clean backups."
                    }
                ],
                "HIGH": [
                    {
                        "title": "Suspicious Network Activity",
                        "template": "Unusual outbound traffic detected from {resource} to known malicious IP {ip}. Traffic volume: {traffic_volume}GB. Duration: {duration} minutes. Protocol: {protocol}.",
                        "impact": "Potential data exfiltration or command and control communication.",
                        "recommended_action": "Block suspicious IPs, analyze traffic patterns, scan for malware."
                    }
                ],
                "MEDIUM": [
                    {
                        "title": "SSL Certificate Expiring",
                        "template": "SSL certificate for {resource} is expiring in {days} days. Certificate details: Issuer: {issuer}, Valid until: {expiry_date}.",
                        "impact": "Service interruption risk if certificate expires.",
                        "recommended_action": "Renew SSL certificate before expiration date."
                    }
                ],
                "LOW": [
                    {
                        "title": "New Resource Deployment",
                        "template": "New resource {resource_type} deployed in subscription {subscription}. Deployed by: {user}. Resource details: {details}",
                        "impact": "Minimal - standard deployment activity.",
                        "recommended_action": "Review deployment for compliance with security policies."
                    }
                ]
            },
            "Performance": {
                "CRITICAL": [
                    {
                        "title": "Database CPU Saturation",
                        "template": "Severe CPU pressure detected on {resource}. Current CPU usage: {cpu}%. Duration: {duration} minutes. Active queries: {query_count}. Top resource-intensive queries identified.",
                        "impact": "Severe performance degradation affecting all database operations.",
                        "recommended_action": "Scale up database resources, optimize top resource-consuming queries."
                    }
                ],
                "HIGH": [
                    {
                        "title": "Memory Pressure Alert",
                        "template": "High memory utilization on {resource}. Memory usage: {memory}%. Available memory: {available_memory}GB. Paging activity detected. Top memory-consuming processes: {processes}.",
                        "impact": "Application performance degradation and increased response times.",
                        "recommended_action": "Increase memory allocation, investigate memory leaks."
                    }
                ],
                "MEDIUM": [
                    {
                        "title": "Increased API Latency",
                        "template": "API endpoint {endpoint} showing increased latency. Average response time: {latency}ms (threshold: {threshold}ms). Affected operations: {operations}.",
                        "impact": "Moderate performance impact on API consumers.",
                        "recommended_action": "Review recent changes, check backend dependencies."
                    }
                ],
                "LOW": [
                    {
                        "title": "Disk Space Warning",
                        "template": "Storage space warning for {resource}. Used space: {used}%. Growth rate: {growth_rate}% per day. Estimated days until full: {days_remaining}.",
                        "impact": "Potential future storage constraints.",
                        "recommended_action": "Plan storage expansion or cleanup unused data."
                    }
                ]
            },
            "Availability": {
                "CRITICAL": [
                    {
                        "title": "Service Outage",
                        "template": "Complete service outage detected for {service} in region {region}. Error rate: 100%. Duration: {duration} minutes. Affected customers: {customer_count}. Root cause: {root_cause}.",
                        "impact": "Complete service unavailability affecting all customers.",
                        "recommended_action": "Initiate DR protocol, failover to backup region."
                    }
                ],
                "HIGH": [
                    {
                        "title": "Database Connectivity Issues",
                        "template": "Intermittent database connectivity failures detected. Connection timeout rate: {timeout_rate}%. Affected databases: {databases}. Error pattern: {error_pattern}.",
                        "impact": "Frequent transaction failures and application errors.",
                        "recommended_action": "Check network connectivity, database locks, and connection pooling."
                    }
                ],
                "MEDIUM": [
                    {
                        "title": "Degraded Performance",
                        "template": "Service {service} experiencing degraded performance. Success rate: {success_rate}%. Average latency increase: {latency_increase}%. Affected operations: {operations}.",
                        "impact": "Slower response times and occasional timeouts.",
                        "recommended_action": "Scale out service, check resource utilization."
                    }
                ],
                "LOW": [
                    {
                        "title": "Single Node Failure",
                        "template": "Node {node} in cluster {cluster} is unhealthy. Health check failures: {failures}. Duration: {duration} minutes. Cluster health: {health_status}.",
                        "impact": "Minimal - redundant nodes handling traffic.",
                        "recommended_action": "Investigate node health, consider replacement."
                    }
                ]
            }
        }

    def generate_ip(self):
        return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

    def generate_resource_id(self, service, component=None):
        subscription_id = str(uuid.uuid4())
        resource_group = f"rg-{service.lower()}"
        if not component:
            component = random.choice(self.services.get(service, [service]))
        name = f"{service.lower()}-{component}-{random.randint(1,999):03d}"
        return f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.{service}/{name}"

    def generate_incident(self, category=None, severity=None):
        if not category:
            category = random.choice(list(self.incident_patterns.keys()))
        if not severity:
            severity = random.choices(self.severity_levels, weights=self.severity_weights)[0]

        # Get random incident pattern for category and severity
        patterns = self.incident_patterns[category][severity]
        pattern = random.choice(patterns)

        # Generate dynamic values
        values = {
            "ip": self.generate_ip(),
            "resource": self.generate_resource_id("KeyVault"),
            "location": random.choice(["East US", "West Europe", "Southeast Asia"]),
            "attempt_count": random.randint(50, 500),
            "traffic_volume": random.randint(1, 100),
            "duration": random.randint(5, 120),
            "protocol": random.choice(["TCP", "UDP", "HTTP"]),
            "days": random.randint(1, 30),
            "issuer": "DigiCert Inc",
            "expiry_date": (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "cpu": random.randint(90, 100),
            "query_count": random.randint(100, 1000),
            "memory": random.randint(85, 100),
            "available_memory": random.randint(1, 16),
            "processes": "process1, process2, process3",
            "endpoint": "/api/v1/orders",
            "latency": random.randint(500, 2000),
            "threshold": 200,
            "operations": "GET, POST, PUT",
            "used": random.randint(85, 95),
            "growth_rate": random.uniform(0.5, 5.0),
            "days_remaining": random.randint(1, 30),
            "service": random.choice(list(self.services.keys())),
            "region": random.choice(["East US", "West Europe", "Southeast Asia"]),
            "customer_count": random.randint(100, 10000),
            "root_cause": random.choice(["Hardware Failure", "Network Partition", "Configuration Error"]),
            "timeout_rate": random.randint(10, 50),
            "databases": "db1, db2, db3",
            "error_pattern": "Connection timeout",
            "success_rate": random.randint(50, 90),
            "latency_increase": random.randint(50, 200),
            "node": f"node-{random.randint(1,10)}",
            "cluster": f"aks-cluster-{random.randint(1,5)}",
            "failures": random.randint(3, 10),
            "health_status": random.choice(["Warning", "Degraded"])
        }

        # Fill in any missing template values with defaults
        template_vars = [v[1] for v in string.Formatter().parse(pattern["template"]) if v[1] is not None]
        for var in template_vars:
            if var not in values:
                values[var] = f"sample_{var}"

        incident = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "category": category,
            "severity": severity,
            "title": pattern["title"],
            "description": pattern["template"].format(**values),
            "impact": pattern["impact"],
            "recommended_action": pattern["recommended_action"],
            "source": {
                "service": values["service"],
                "region": values["region"],
                "resource_id": self.generate_resource_id(values["service"])
            },
            "metadata": {
                "detection_source": random.choice(["Azure Monitor", "Security Center", "Custom Alert"]),
                "alert_type": pattern["title"].replace(" ", "_").upper(),
                "correlation_id": str(uuid.uuid4())
            }
        }
        return incident

def insert_enhanced_incidents(count=100):
    generator = EnhancedLogGenerator()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Generating and inserting {count} enhanced incidents...")
    for _ in range(count):
        incident = generator.generate_incident()
        cursor.execute("""
            INSERT INTO incident_logs (id, incident_json, source_type, status, severity) 
            VALUES (?, ?, ?, ?, ?)
        """, (
            incident["id"],
            json.dumps(incident),
            incident["metadata"]["detection_source"],
            "new",
            incident["severity"]
        ))
    
    conn.commit()
    conn.close()
    print(f"Successfully inserted {count} enhanced incidents with detailed descriptions.")

def analyze_incident_distribution():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\nIncident Distribution Analysis:")
    
    # Severity distribution
    cursor.execute("""
        SELECT severity, COUNT(*) as count 
        FROM incident_logs 
        GROUP BY severity
    """)
    print("\nSeverity Distribution:")
    for severity, count in cursor.fetchall():
        print(f"{severity}: {count} incidents")
    
    # Category distribution
    cursor.execute("""
        SELECT 
            json_extract(incident_json, '$.category') as category,
            COUNT(*) as count
        FROM incident_logs 
        GROUP BY json_extract(incident_json, '$.category')
    """)
    print("\nCategory Distribution:")
    for category, count in cursor.fetchall():
        print(f"{category}: {count} incidents")
    
    conn.close()

if __name__ == '__main__':
    # Generate substantial dataset
    insert_enhanced_incidents(200)  # Generate 200 diverse incidents
    analyze_incident_distribution()