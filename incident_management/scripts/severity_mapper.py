import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import torch

# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

class SeverityMapper:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Severity rules with patterns and weights
        self.severity_rules = {
            "critical": {
                "patterns": [
                    "system down",
                    "service unavailable",
                    "data loss",
                    "security breach",
                    "critical failure",
                    "system crash",
                    "complete outage",
                    "data corruption",
                    "ssl certificate expired",
                    "authentication failure",
                    "database corruption",
                    "memory exhaustion",
                    "deadlock detected"
                ],
                "weight": 1.0
            },
            "high": {
                "patterns": [
                    "performance degradation",
                    "high latency",
                    "partial outage",
                    "error rate increase",
                    "resource exhaustion",
                    "api failure",
                    "database performance",
                    "memory leak",
                    "network connectivity",
                    "threshold breach"
                ],
                "weight": 0.8
            },
            "medium": {
                "patterns": [
                    "warning",
                    "slow response",
                    "minor disruption",
                    "intermittent issue",
                    "resource pressure",
                    "configuration change",
                    "service degradation"
                ],
                "weight": 0.6
            },
            "low": {
                "patterns": [
                    "informational",
                    "routine",
                    "scheduled maintenance",
                    "minor issue",
                    "update completed",
                    "automatic recovery"
                ],
                "weight": 0.4
            }
        }
        
        # Pre-compute embeddings for all patterns
        self.pattern_embeddings = {}
        for severity, data in self.severity_rules.items():
            self.pattern_embeddings[severity] = {
                pattern: self.model.encode(pattern.lower())
                for pattern in data["patterns"]
            }

    def initialize_db(self):
        """Create the severity_mappings table if it doesn't exist"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS severity_mappings (
            incident_id TEXT PRIMARY KEY,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            rule_match TEXT,
            bert_score REAL,
            combined_score REAL,
            FOREIGN KEY (incident_id) REFERENCES incident_logs(id)
        )
        """)
        
        conn.commit()
        conn.close()

    def get_semantic_similarity(self, text, pattern_embedding):
        """Calculate semantic similarity using BERT embeddings"""
        text_embedding = self.model.encode(text.lower())
        return 1 - cosine(text_embedding, pattern_embedding)

    def analyze_text(self, text):
        """Analyze text using both rule-based and BERT-based approaches"""
        best_match = {
            "severity": "low",
            "confidence": 0.0,
            "rule_match": None,
            "bert_score": 0.0
        }

        for severity, data in self.severity_rules.items():
            # Check for exact matches first
            for pattern in data["patterns"]:
                if pattern.lower() in text.lower():
                    return {
                        "severity": severity,
                        "confidence": 1.0,
                        "rule_match": pattern,
                        "bert_score": 1.0
                    }

            # Calculate BERT similarity for each pattern
            max_similarity = 0.0
            best_pattern = None
            
            for pattern, pattern_embedding in self.pattern_embeddings[severity].items():
                similarity = self.get_semantic_similarity(text, pattern_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern = pattern

            # Calculate combined score using weights
            combined_score = max_similarity * data["weight"]
            
            if combined_score > best_match["confidence"]:
                best_match = {
                    "severity": severity,
                    "confidence": combined_score,
                    "rule_match": best_pattern,
                    "bert_score": max_similarity
                }

        return best_match

    def map_incidents(self):
        """Map severities for all incidents in the database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all incidents that don't have severity mappings
        cursor.execute("""
        SELECT il.id, il.incident_json 
        FROM incident_logs il 
        LEFT JOIN severity_mappings sm ON il.id = sm.incident_id
        WHERE sm.incident_id IS NULL
        """)
        
        incidents = cursor.fetchall()
        print(f"Processing {len(incidents)} incidents...")

        for incident_id, incident_json in incidents:
            incident_data = json.loads(incident_json)
            
            # Extract relevant text for analysis
            analysis_text = ""
            
            # Add error message if present
            if isinstance(incident_data, dict):
                # Extract error message if present
                if "properties" in incident_data and isinstance(incident_data["properties"], dict):
                    if "error" in incident_data["properties"] and isinstance(incident_data["properties"]["error"], dict):
                        analysis_text += str(incident_data["properties"]["error"].get("message", "")) + " "
                
                # Add operation name
                if "operationName" in incident_data:
                    if isinstance(incident_data["operationName"], dict):
                        analysis_text += str(incident_data["operationName"].get("value", "")) + " "
                    else:
                        analysis_text += str(incident_data["operationName"]) + " "
                
                # Add status information
                if "status" in incident_data:
                    if isinstance(incident_data["status"], dict):
                        analysis_text += str(incident_data["status"].get("value", "")) + " "
                    else:
                        analysis_text += str(incident_data["status"]) + " "

            # Analyze the text
            severity_info = self.analyze_text(analysis_text.strip())
            
            # Insert the severity mapping
            cursor.execute("""
            INSERT INTO severity_mappings 
            (incident_id, severity, confidence, rule_match, bert_score, combined_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                incident_id,
                severity_info["severity"],
                severity_info["confidence"],
                severity_info["rule_match"],
                severity_info["bert_score"],
                severity_info["confidence"]  # combined_score is same as confidence in this case
            ))

            if cursor.rowcount % 100 == 0:
                conn.commit()
                print(f"Processed {cursor.rowcount} incidents...")

        conn.commit()
        
        # Print summary
        cursor.execute("""
        SELECT severity, COUNT(*) as count, 
               AVG(confidence) as avg_confidence,
               AVG(bert_score) as avg_bert_score
        FROM severity_mappings
        GROUP BY severity
        ORDER BY avg_confidence DESC
        """)
        
        print("\nSeverity Distribution Summary:")
        for severity, count, avg_confidence, avg_bert_score in cursor.fetchall():
            print(f"{severity.upper()}: Count={count}, "
                  f"Avg Confidence={avg_confidence:.2f}, "
                  f"Avg BERT Score={avg_bert_score:.2f}")

        conn.close()

if __name__ == "__main__":
    mapper = SeverityMapper()
    mapper.initialize_db()
    mapper.map_incidents()