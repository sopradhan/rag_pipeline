import sqlite3
import json
import uuid
import random
from datetime import datetime, timedelta
import ipaddress

DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

class AzureLogGenerator:
    def __init__(self):
        # Environment-specific subscriptions (typical enterprise setup)
        self.environments = {
            "prod": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "contoso-prod",
                "tags": {
                    "Environment": "Production",
                    "CostCenter": "IT-PROD-001",
                    "Criticality": "High",
                    "DataClassification": "Confidential"
                }
            },
            "uat": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "contoso-uat",
                "tags": {
                    "Environment": "UAT",
                    "CostCenter": "IT-TEST-001",
                    "Criticality": "Medium",
                    "DataClassification": "Internal"
                }
            },
            "dev": {
                "subscription_id": str(uuid.uuid4()),
                "subscription_name": "contoso-dev",
                "tags": {
                    "Environment": "Development",
                    "CostCenter": "IT-DEV-001",
                    "Criticality": "Low",
                    "DataClassification": "Internal"
                }
            }
        }
        
        # Azure regions by environment
        self.locations = {
            "prod": ["eastus", "westeurope", "southeastasia"],  # Primary regions
            "uat": ["eastus2", "northeurope"],                  # Secondary regions
            "dev": ["westus2", "centralus"]                     # Dev regions
        }
        
        # Resource group naming patterns by environment
        self.resource_groups = {
            "prod": [
                "rg-prod-core-shared",
                "rg-prod-app-primary",
                "rg-prod-data-primary",
                "rg-prod-monitoring",
                "rg-prod-security"
            ],
            "uat": [
                "rg-uat-shared",
                "rg-uat-apps",
                "rg-uat-data",
                "rg-uat-test"
            ],
            "dev": [
                "rg-dev-shared",
                "rg-dev-apps",
                "rg-dev-sandbox"
            ]
        }
        
        # Azure service configuration
        self.services = {
            "Microsoft.KeyVault": {
                "type": "vaults",
                "operations": ["VaultGet", "KeyGet", "KeyCreate", "SecretGet", "SecretSet"],
                "resultTypes": ["Success", "Failed"],
                "statusCodes": [200, 201, 403, 404, 500]
            },
            "Microsoft.Sql": {
                "type": "servers/databases",
                "operations": ["DatabaseConnect", "QueryExecute", "BackupComplete", "DatabaseFailover"],
                "resultTypes": ["Succeeded", "Failed"],
                "statusCodes": [200, 404, 408, 500]  # 408 is timeout
            },
            "Microsoft.Web": {
                "type": "sites",
                "operations": ["AppServicePlanUpdate", "WebAppRestart", "SiteConfigUpdate"],
                "resultTypes": ["Succeeded", "Failed", "InProgress"],
                "statusCodes": [200, 202, 404, 500]
            },
            "Microsoft.Storage": {
                "type": "storageAccounts",
                "operations": ["BlobGet", "BlobCreate", "ContainerDelete", "StorageRead"],
                "resultTypes": ["Success", "Failed"],
                "statusCodes": [200, 201, 404, 500]
            },
            "Microsoft.Network": {
                "type": "virtualNetworks",
                "operations": ["NetworkSecurityGroupCreate", "SubnetUpdate", "RouteTableUpdate"],
                "resultTypes": ["Succeeded", "Failed"],
                "statusCodes": [200, 201, 400, 500]
            }
        }
        
        # Caller types and roles
        self.callers = {
            "user": ["admin@contoso.com", "developer@contoso.com", "ops@contoso.com"],
            "principal": ["app-service-principal", "aks-cluster-principal", "automation-account"],
            "system": ["Microsoft.Azure.Management", "Microsoft.Azure.Monitor", "Microsoft.Azure.Automation"]
        }
        
        # Common error patterns
        self.error_patterns = {
            "authorization": {
                "code": "AuthorizationFailed",
                "message": "Client '{0}' with object id '{1}' does not have authorization to perform action"
            },
            "throttling": {
                "code": "RequestThrottled",
                "message": "Request was throttled. Please retry after {0} seconds"
            },
            "notfound": {
                "code": "ResourceNotFound",
                "message": "Resource '{0}' was not found"
            }
        }

    def generate_resource_id(self, service, env=None):
        if not env:
            env = random.choice(["prod", "uat", "dev"])
        
        # Get environment-specific details
        sub_info = self.environments[env]
        rg = random.choice(self.resource_groups[env])
        location = random.choice(self.locations[env])
        svc_config = self.services[service]
        
        # Generate resource name following environment conventions
        resource_type = svc_config['type'].split('/')[0]
        if env == "prod":
            name = f"{resource_type}-{location}-{random.randint(1,99):02d}"
        else:
            name = f"{resource_type}-{env}-{location}-{random.randint(1,99):02d}"
            
        return {
            "id": f"/subscriptions/{sub_info['subscription_id']}/resourceGroups/{rg}/providers/{service}/{svc_config['type']}/{name}",
            "name": name,
            "location": location,
            "subscription": sub_info['subscription_name'],
            "resourceGroup": rg,
            "tags": {
                **sub_info['tags'],
                "Application": f"App-{random.randint(1,10):02d}",
                "Owner": "team-" + ("prod" if env == "prod" else "nonp"),
                "Location": location,
                "Service": service.split('.')[-1]
            }
        }

    def generate_correlation_id(self):
        return str(uuid.uuid4())

    def generate_activity_log(self):
        # Select environment based on weighted distribution
        env_weights = {"prod": 0.5, "uat": 0.3, "dev": 0.2}  # 50% prod, 30% uat, 20% dev
        env = random.choices(list(env_weights.keys()), 
                           weights=list(env_weights.values()))[0]
        
        service = random.choice(list(self.services.keys()))
        svc_config = self.services[service]
        operation = random.choice(svc_config['operations'])
        status = random.choice(svc_config['resultTypes'])
        
        # Get full resource context including tags
        resource = self.generate_resource_id(service, env)
        
        # Determine caller type and identity based on environment
        if env == "prod":
            caller_type = random.choice(["system", "principal"])  # No direct user access in prod
        else:
            caller_type = random.choice(list(self.callers.keys()))
        
        timestamp = (datetime.utcnow() - timedelta(minutes=random.randint(0, 1440))).isoformat() + "Z"
        
        log = {
            "authorization": {
                "scope": resource["id"],
                "action": f"{service}/{operation}",
                "evidence": {
                    "role": "Contributor",
                    "roleAssignmentScope": resource["id"],
                    "roleAssignmentId": str(uuid.uuid4())
                }
            },
            "caller": self.callers[caller_type][0],  # Use first caller from the type
            "correlationId": self.generate_correlation_id(),
            "eventTimestamp": timestamp,
            "level": "Informational" if status in ["Success", "Succeeded"] else "Error",
            "operationName": {
                "value": operation,
                "localizedValue": operation
            },
            "resourceId": resource["id"],
            "status": {
                "value": status,
                "localizedValue": status
            },
            "subscriptionId": resource["id"].split('/')[2],
            "tags": resource["tags"],
            "tenantId": str(uuid.uuid4()),
            "properties": {
                "statusCode": random.choice(svc_config['statusCodes']),
                "serviceRequestId": str(uuid.uuid4()),
                "resourceType": f"{service}/{svc_config['type']}"
            }
        }
        
        if status in ["Failed", "Error"]:
            error = random.choice(list(self.error_patterns.values()))
            log["properties"]["error"] = {
                "code": error["code"],
                "message": error["message"].format(self.callers[caller_type][0], str(uuid.uuid4()))
            }
        
        return log

    def generate_metric_log(self):
        # Select environment with weighting
        env_weights = {"prod": 0.6, "uat": 0.25, "dev": 0.15}  # More monitoring in prod
        env = random.choices(list(env_weights.keys()), 
                           weights=list(env_weights.values()))[0]
        
        service = random.choice(list(self.services.keys()))
        resource = self.generate_resource_id(service, env)
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        metrics = {
            "Microsoft.KeyVault": {
                "name": "ServiceApiLatency",
                "value": random.uniform(10, 1000),
                "unit": "Milliseconds"
            },
            "Microsoft.Sql": {
                "name": "cpu_percent",
                "value": random.uniform(0, 100),
                "unit": "Percent"
            },
            "Microsoft.Web": {
                "name": "Http5xx",
                "value": random.randint(0, 100),
                "unit": "Count"
            },
            "Microsoft.Storage": {
                "name": "Transactions",
                "value": random.randint(100, 10000),
                "unit": "Count"
            },
            "Microsoft.Network": {
                "name": "BytesTransmittedRate",
                "value": random.uniform(1000, 100000),
                "unit": "BytesPerSecond"
            }
        }
        
        metric = metrics[service]
        return {
            "time": timestamp,
            "resourceId": resource["id"],
            "metricName": metric["name"],
            "timeGrain": "PT1M",
            "value": metric["value"],
            "unit": metric["unit"],
            "dimensions": {
                "location": resource["location"],
                "apiName": random.choice(self.services[service]["operations"])
            }
        }

    def generate_diagnostic_log(self):
        service = random.choice(list(self.services.keys()))
        svc_config = self.services[service]
        resource_id = self.generate_resource_id(service)
        operation = random.choice(svc_config['operations'])
        status_code = random.choice(svc_config['statusCodes'])
        
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        log = {
            "time": timestamp,
            "resourceId": resource_id,
            "operationName": operation,
            "category": "ResourceHealth",
            "resultType": "Success" if status_code < 400 else "Failed",
            "correlationId": self.generate_correlation_id(),
            "properties": {
                "statusCode": status_code,
                "serviceRequestId": str(uuid.uuid4()),
                "statusMessage": f"Operation {operation} completed with status {status_code}"
            }
        }
        
        return log

def insert_realistic_logs(count=1000):
    generator = AzureLogGenerator()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM incident_logs")
    
    print(f"Generating and inserting {count} realistic Azure logs...")
    for i in range(count):
        # Generate different types of logs with distribution
        if i % 3 == 0:
            log = generator.generate_activity_log()
            source_type = "ActivityLog"
        elif i % 3 == 1:
            log = generator.generate_metric_log()
            source_type = "MetricLog"
        else:
            log = generator.generate_diagnostic_log()
            source_type = "DiagnosticLog"
            
        cursor.execute("""
            INSERT INTO incident_logs (id, incident_json, source_type, status)
            VALUES (?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            json.dumps(log),
            source_type,
            "new"
        ))
        
        if (i + 1) % 100 == 0:
            print(f"Inserted {i + 1} logs...")
    
    conn.commit()
    
    # Print summary
    cursor.execute("SELECT source_type, COUNT(*) FROM incident_logs GROUP BY source_type")
    distribution = cursor.fetchall()
    print("\nLog Distribution:")
    for source_type, count in distribution:
        print(f"{source_type}: {count} logs")
    
    conn.close()
    print("\nLog generation completed!")

if __name__ == "__main__":
    insert_realistic_logs(1000)