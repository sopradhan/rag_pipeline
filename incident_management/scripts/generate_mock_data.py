import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

class MockDataGenerator:
    def __init__(self):
        self.incident_types = [
            "Server Outage",
            "Database Performance",
            "Network Latency",
            "Security Breach",
            "API Error",
            "Memory Leak",
            "CPU Spike",
            "Disk Space",
            "Service Crash",
            "Authentication Issue"
        ]
        
        self.departments = [
            "Infrastructure",
            "Database",
            "Network",
            "Security",
            "Application",
            "DevOps"
        ]
        
        self.root_causes = {
            "Server Outage": [
                "Hardware failure",
                "Power issue",
                "Resource exhaustion",
                "OS crash"
            ],
            "Database Performance": [
                "Query optimization needed",
                "Index fragmentation",
                "Resource contention",
                "Memory pressure"
            ],
            "Network Latency": [
                "Bandwidth saturation",
                "DNS issues",
                "Routing problems",
                "Network congestion"
            ],
            "Security Breach": [
                "Unauthorized access",
                "SQL injection",
                "Malware detected",
                "Credential compromise"
            ]
        }
        
        self.corrective_actions = {
            "Hardware failure": [
                "Replace faulty hardware",
                "Perform hardware diagnostics",
                "Update firmware"
            ],
            "Query optimization needed": [
                "Optimize SQL queries",
                "Update indexes",
                "Review execution plans"
            ],
            "Bandwidth saturation": [
                "Increase bandwidth",
                "Implement traffic shaping",
                "Optimize network routes"
            ],
            "Unauthorized access": [
                "Reset credentials",
                "Enable 2FA",
                "Review access logs"
            ]
        }

    def generate_incident(self, incident_id: int) -> Dict[str, Any]:
        incident_type = random.choice(self.incident_types)
        department = random.choice(self.departments)
        
        # Generate timestamp between now and 30 days ago
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Select root cause
        root_causes = self.root_causes.get(incident_type, ["Unknown issue"])
        root_cause = random.choice(root_causes)
        
        # Select corrective actions
        corrective_actions = self.corrective_actions.get(root_cause, ["Investigate and resolve"])
        corrective_action = random.choice(corrective_actions)
        
        # Generate severity based on weighted random choice
        severity = random.choices(
            ["LOW", "MEDIUM", "HIGH"],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        # Generate resolution time based on severity
        resolution_hours = {
            "LOW": random.uniform(1, 4),
            "MEDIUM": random.uniform(4, 12),
            "HIGH": random.uniform(12, 48)
        }[severity]
        
        resolved_at = timestamp + timedelta(hours=resolution_hours)
        
        return {
            "id": incident_id,
            "type": incident_type,
            "department": department,
            "severity": severity,
            "description": f"{severity} severity {incident_type} detected in {department} department. {root_cause} identified as root cause.",
            "root_cause": root_cause,
            "corrective_action": corrective_action,
            "created_at": timestamp.isoformat(),
            "resolved_at": resolved_at.isoformat(),
            "resolution_time_hours": resolution_hours,
            "status": "resolved",
            "metadata": {
                "system": f"{department.lower()}_system",
                "component": incident_type.lower().replace(" ", "_"),
                "impact_level": severity.lower()
            }
        }

    def generate_incidents(self, num_incidents: int) -> List[Dict[str, Any]]:
        return [self.generate_incident(i) for i in range(num_incidents)]

    def save_to_json(self, incidents: List[Dict[str, Any]], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(incidents, f, indent=2)

    def save_to_csv(self, incidents: List[Dict[str, Any]], filepath: str):
        df = pd.DataFrame(incidents)
        df.to_csv(filepath, index=False)

def main():
    # Initialize generator
    generator = MockDataGenerator()
    
    # Generate 1000 incidents
    incidents = generator.generate_incidents(1000)
    
    # Save to both JSON and CSV formats
    generator.save_to_json(incidents, 'data/mock_incidents.json')
    generator.save_to_csv(incidents, 'data/mock_incidents.csv')
    
    print(f"Generated {len(incidents)} mock incidents")
    
    # Print sample incident
    print("\nSample incident:")
    print(json.dumps(incidents[0], indent=2))

if __name__ == "__main__":
    main()