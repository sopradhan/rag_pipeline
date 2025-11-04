import random
from datetime import datetime, timedelta
import json
import uuid

class CategoryDataGenerator:
    def __init__(self):
        # Domain Categories
        self.domain_categories = {
            "Security": ["Access Control", "Authentication", "Data Privacy", "Network Security", "Compliance"],
            "Infrastructure": ["Servers", "Networks", "Storage", "Cloud", "On-Premise"],
            "Applications": ["Web Apps", "Mobile Apps", "Desktop Apps", "APIs", "Microservices"],
            "Data": ["Databases", "Data Lakes", "Data Warehouses", "ETL", "Analytics"],
            "Operations": ["Monitoring", "Deployment", "Maintenance", "Backup", "Recovery"]
        }

        # Line of Business Categories
        self.lob_categories = {
            "Finance": ["Banking", "Insurance", "Investment", "Accounting", "Treasury"],
            "Healthcare": ["Patient Care", "Medical Records", "Insurance Claims", "Pharmacy", "Lab Services"],
            "Retail": ["E-commerce", "Point of Sale", "Inventory", "Supply Chain", "Customer Service"],
            "Manufacturing": ["Production", "Quality Control", "Supply Chain", "Assembly", "Logistics"],
            "Technology": ["Software Development", "IT Services", "Cloud Services", "Consulting", "Support"]
        }

        # Technical Categories
        self.technical_categories = {
            "Frontend": ["UI/UX", "React", "Angular", "Vue", "Mobile"],
            "Backend": ["APIs", "Databases", "Microservices", "Authentication", "Caching"],
            "Infrastructure": ["AWS", "Azure", "GCP", "Kubernetes", "Docker"],
            "Data": ["SQL", "NoSQL", "ETL", "Data Lakes", "Analytics"],
            "Security": ["IAM", "Encryption", "Network Security", "Compliance", "Auditing"]
        }

        # Severity Levels
        self.severity_levels = ["Low", "Medium", "High", "Critical"]

        # Status Types
        self.status_types = ["Open", "In Progress", "Under Review", "Resolved", "Closed"]

        # Priority Levels
        self.priority_levels = ["P1", "P2", "P3", "P4"]

    def generate_timestamp(self, start_date=None, end_date=None):
        """Generate a random timestamp between start and end date"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        time_diff = end_date - start_date
        random_days = random.randint(0, time_diff.days)
        random_seconds = random.randint(0, 24*60*60)
        return start_date + timedelta(days=random_days, seconds=random_seconds)

    def generate_single_record(self):
        """Generate a single record with domain, LOB, and technical categories"""
        # Select random categories
        domain_main = random.choice(list(self.domain_categories.keys()))
        domain_sub = random.choice(self.domain_categories[domain_main])
        
        lob_main = random.choice(list(self.lob_categories.keys()))
        lob_sub = random.choice(self.lob_categories[lob_main])
        
        tech_main = random.choice(list(self.technical_categories.keys()))
        tech_sub = random.choice(self.technical_categories[tech_main])

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": self.generate_timestamp().isoformat(),
            "domain": {
                "main_category": domain_main,
                "sub_category": domain_sub,
                "confidence_score": round(random.uniform(0.7, 1.0), 2)
            },
            "line_of_business": {
                "main_category": lob_main,
                "sub_category": lob_sub,
                "confidence_score": round(random.uniform(0.7, 1.0), 2)
            },
            "technical": {
                "main_category": tech_main,
                "sub_category": tech_sub,
                "confidence_score": round(random.uniform(0.7, 1.0), 2)
            },
            "metadata": {
                "severity": random.choice(self.severity_levels),
                "priority": random.choice(self.priority_levels),
                "status": random.choice(self.status_types),
                "assigned_team": f"Team-{random.randint(1,5)}",
                "resolution_time": random.randint(1, 72) if random.random() > 0.3 else None
            },
            "description": self._generate_description(domain_main, lob_main, tech_main)
        }
        return record

    def _generate_description(self, domain, lob, tech):
        """Generate a realistic description based on the categories"""
        templates = [
            f"Issue reported in {domain} system affecting {lob} operations. Technical investigation reveals {tech}-related root cause.",
            f"{lob} team reported degraded performance in {domain} components. {tech} team investigating the underlying cause.",
            f"Critical alert from {domain} monitoring system impacting {lob} services. Requires immediate {tech} team intervention.",
            f"Scheduled maintenance required for {domain} infrastructure supporting {lob} applications. {tech} team to coordinate the effort.",
            f"Security audit identified potential risks in {domain} systems used by {lob}. {tech} team implementing recommended fixes."
        ]
        return random.choice(templates)

    def generate_dataset(self, num_records=100, output_file="category_data.json"):
        """Generate multiple records and save to a JSON file"""
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "record_count": num_records,
                "version": "1.0"
            },
            "records": [self.generate_single_record() for _ in range(num_records)]
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset

def main():
    generator = CategoryDataGenerator()
    dataset = generator.generate_dataset(num_records=100)
    print(f"Generated {len(dataset['records'])} records")
    print("Sample record:")
    print(json.dumps(dataset['records'][0], indent=2))

if __name__ == "__main__":
    main()