import random
from datetime import datetime, timedelta
import json
import uuid
import ipaddress
import hashlib

class SystemMetricsGenerator:
    def __init__(self):
        # Azure Service Components
        self.azure_services = {
            "Key Vault": {
                "metrics": ["AvailabilityResults", "ServiceApiLatency", "SaturationShoebox", "Authentication"],
                "operations": ["Secret Get", "Secret List", "Key Sign", "Key Verify", "Certificate Get"],
                "alert_types": ["Access Denied", "Crypto Operations Failed", "Certificate Expiry", "Network Issues"]
            },
            "App Service": {
                "metrics": ["Http5xx", "ResponseTime", "CpuTime", "MemoryWorkingSet"],
                "operations": ["HTTP GET", "HTTP POST", "WebSocket", "Custom Domain SSL"],
                "alert_types": ["High Memory Usage", "HTTP Server Errors", "SSL Certificate Expiry"]
            },
            "SQL Database": {
                "metrics": ["cpu_percent", "deadlock", "dtu_consumption_percent", "storage_percent"],
                "operations": ["Database CRUD", "Backup", "Scale", "Failover"],
                "alert_types": ["High CPU Usage", "Storage Space Low", "Deadlock Detected", "Backup Failed"]
            }
        }

        # Common Alert Patterns
        self.alert_patterns = {
            "Security": {
                "high": [
                    "Unauthorized access attempt detected from IP {ip}",
                    "Multiple failed authentication attempts for Key Vault {resource}",
                    "Suspicious certificate operation detected in {resource}",
                    "Potential data exfiltration attempt from {resource}"
                ],
                "medium": [
                    "SSL certificate expiring soon for {resource}",
                    "Unusual access pattern detected in {resource}",
                    "Service principal credentials expiring for {resource}"
                ],
                "low": [
                    "New IP address accessing {resource}",
                    "Minor permission changes in {resource}",
                    "Routine security scan alerts for {resource}"
                ]
            },
            "Performance": {
                "high": [
                    "Critical performance degradation in {resource}",
                    "Service unavailable - {resource}",
                    "Database deadlock detected in {resource}"
                ],
                "medium": [
                    "High CPU utilization in {resource}",
                    "Memory pressure detected in {resource}",
                    "Increased latency in {resource}"
                ],
                "low": [
                    "Minor performance deviation in {resource}",
                    "Slightly elevated response times in {resource}",
                    "Routine maintenance impact on {resource}"
                ]
            },
            "System": {
                "high": [
                    "System crash detected in {resource}",
                    "Data corruption identified in {resource}",
                    "Critical system component failure in {resource}"
                ],
                "medium": [
                    "System warnings from {resource}",
                    "Component degradation in {resource}",
                    "Backup delay detected for {resource}"
                ],
                "low": [
                    "System notification from {resource}",
                    "Minor configuration drift in {resource}",
                    "Routine system alert from {resource}"
                ]
            }
        }

        # Resource Regions
        self.regions = ["eastus", "westus2", "centralus", "northeurope", "westeurope", "southeastasia"]
        
        # Alert Categories
        self.categories = ["Security", "Performance", "System"]
        
        # Severity Levels with Weights
        self.severity_weights = {
            "high": 0.2,     # 20% chance
            "medium": 0.3,   # 30% chance
            "low": 0.5       # 50% chance
        }

    def generate_resource_id(self, service, region):
        """Generate a realistic Azure resource ID"""
        subscription_id = f"{uuid.uuid4()}"
        resource_group = f"rg-{service.lower()}-{region}"
        resource_name = f"{service.lower()}-{region}-{random.randint(1, 999):03d}"
        return f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.{service.replace(' ', '')}/vaults/{resource_name}"

    def generate_correlation_id(self):
        """Generate a correlation ID for tracking related incidents"""
        return hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:12]

    def generate_metric_data(self, service, metric):
        """Generate realistic metric data based on service and metric type"""
        base_value = random.uniform(50, 95)  # Base performance metric
        jitter = random.uniform(-5, 5)       # Add some randomness
        return round(base_value + jitter, 2)

    def generate_ip_address(self):
        """Generate a random IP address"""
        return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))

    def generate_incident(self):
        """Generate a single incident log entry"""
        # Select random service and region
        service = random.choice(list(self.azure_services.keys()))
        region = random.choice(self.regions)
        category = random.choice(self.categories)
        severity = random.choices(list(self.severity_weights.keys()), 
                                weights=list(self.severity_weights.values()))[0]
        
        # Generate timestamps
        detected_time = datetime.now() - timedelta(
            minutes=random.randint(0, 60),
            seconds=random.randint(0, 60)
        )
        
        resolution_time = None
        status = "Active"
        if random.random() > 0.3:  # 70% chance of being resolved
            resolution_minutes = random.randint(5, 180)
            resolution_time = detected_time + timedelta(minutes=resolution_minutes)
            status = "Resolved"

        # Generate incident details
        resource_id = self.generate_resource_id(service, region)
        correlation_id = self.generate_correlation_id()
        
        incident = {
            "incident_id": str(uuid.uuid4()),
            "correlation_id": correlation_id,
            "resource_id": resource_id,
            "service": service,
            "region": region,
            "category": category,
            "severity": severity,
            "status": status,
            "detected_time": detected_time.isoformat(),
            "resolution_time": resolution_time.isoformat() if resolution_time else None,
            "alert": {
                "type": random.choice(self.azure_services[service]["alert_types"]),
                "message": random.choice(self.alert_patterns[category][severity.lower()]).format(
                    resource=resource_id.split('/')[-1],
                    ip=self.generate_ip_address()
                ),
                "metric_name": random.choice(self.azure_services[service]["metrics"]),
                "metric_value": self.generate_metric_data(service, "metric"),
                "threshold": 95.0
            },
            "metadata": {
                "operation": random.choice(self.azure_services[service]["operations"]),
                "source_ip": self.generate_ip_address(),
                "user_agent": "Azure-Security-Center/1.0",
                "subscription_id": resource_id.split('/')[2],
                "resource_group": resource_id.split('/')[4]
            },
            "actions_taken": self._generate_actions(status, service, category)
        }
        return incident

    def _generate_actions(self, status, service, category):
        """Generate a list of actions taken for the incident"""
        actions = [{
            "timestamp": datetime.now().isoformat(),
            "action": "Alert Generated",
            "actor": "Azure Monitor",
            "details": "Incident detected and logged"
        }]
        
        if status == "Resolved":
            actions.extend([
                {
                    "timestamp": (datetime.now() + timedelta(minutes=5)).isoformat(),
                    "action": "Investigation Started",
                    "actor": f"Security Team",
                    "details": f"Investigating {category.lower()} incident in {service}"
                },
                {
                    "timestamp": (datetime.now() + timedelta(minutes=30)).isoformat(),
                    "action": "Resolution",
                    "actor": f"System Admin",
                    "details": "Incident resolved - Normal operation restored"
                }
            ])
        return actions

    def generate_dataset(self, num_records=50, output_file="system_incidents.json"):
        """Generate multiple incident records and save to a JSON file"""
        incidents = []
        for _ in range(num_records):
            incidents.append(self.generate_incident())
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "record_count": len(incidents),
                "version": "1.0"
            },
            "incidents": incidents
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset

def main():
    generator = SystemMetricsGenerator()
    dataset = generator.generate_dataset()
    print(f"Generated {len(dataset['incidents'])} incident records")
    print("\nSample incident:")
    print(json.dumps(dataset['incidents'][0], indent=2))

if __name__ == "__main__":
    main()