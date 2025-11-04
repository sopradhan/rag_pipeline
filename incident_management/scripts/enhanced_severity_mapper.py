import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import torch

# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

class EnhancedSeverityMapper:
    def __init__(self):
        # Initialize BERT model with better error handling
        try:
            # Try loading distilbert-base first (faster and good for technical text)
            self.model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
            print("Using distilbert-base model for semantic analysis")
        except Exception as e1:
            try:
                # Fallback to all-mpnet-base (more accurate but slower)
                self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                print("Using all-mpnet-base model for semantic analysis")
            except Exception as e2:
                # Final fallback to MiniLM (lightweight but still effective)
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("Using MiniLM model for semantic analysis")
        
        self.load_rules()
        self.similarity_threshold = 0.65  # Increased threshold for better precision
        
        # Load trained models if available
        try:
            import joblib
            self.scaler = joblib.load('incident_severity_scaler.joblib')
            self.classifier = joblib.load('incident_severity_classifier.joblib')
            with open('severity_weights.json', 'r') as f:
                self.weights = json.load(f)
            self.use_ml = True
            print("Using ML-based severity classification")
        except Exception as e:
            print(f"ML models not found, using enhanced rule-based classification: {str(e)}")
            self.use_ml = False
            # Default weights for rule-based approach
            self.weights = {
                'bert_weight': 0.35,    # Reduced BERT weight
                'rule_weight': 0.45,    # Increased rule weight
                'env_weight': 0.20      # Added environment weight
            }
        
    def load_rules(self):
        """Load severity rules from the database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sr.pattern, sr.severity_level, sr.base_score, sr.category, sr.description 
            FROM severity_rules sr
        """)
        self.rules = cursor.fetchall()
        
        # Pre-compute embeddings for patterns and store full rule info
        self.pattern_info = {}
        for pattern, sev_level, base_score, category, description in self.rules:
            embedding = self.model.encode(pattern.lower())
            self.pattern_info[pattern] = {
                'embedding': embedding,
                'severity_level': sev_level,
                'base_score': base_score,
                'category': category,
                'description': description
            }
        conn.close()

    def initialize_db(self):
        """Create the enhanced_severity_mappings table"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS enhanced_severity_mappings (
            incident_id TEXT PRIMARY KEY,
            severity_level TEXT NOT NULL,     -- S1, S2, S3, S4
            bert_score REAL NOT NULL,         -- Raw BERT similarity score (0-1)
            rule_score REAL NOT NULL,         -- Base score from rules table (1-100)
            combined_score REAL NOT NULL,      -- Weighted combination (1-100)
            matched_pattern TEXT,
            source_json TEXT,                 -- Original incident JSON
            FOREIGN KEY (incident_id) REFERENCES incident_logs(id)
        )
        """)
        
        conn.commit()
        conn.close()

    def get_semantic_similarity(self, text, pattern_embedding):
        """Calculate semantic similarity using BERT embeddings"""
        text_embedding = self.model.encode(text.lower())
        return 1 - cosine(text_embedding, pattern_embedding)

    def get_environment_factor(self, incident_data):
        """Calculate environment-based severity adjustment"""
        if isinstance(incident_data, dict):
            # Try different possible locations of environment info
            env = None
            if "properties" in incident_data:
                props = incident_data["properties"]
                if isinstance(props, dict):
                    env = props.get("environment", "").lower()
                    if not env:
                        env = props.get("env", "").lower()
            
            if not env and "tags" in incident_data:
                tags = incident_data["tags"]
                if isinstance(tags, dict):
                    env = tags.get("Environment", "").lower()
            
            if not env and "resourceId" in incident_data:
                # Try to extract from resource ID (e.g., /subscriptions/.../prod-...)
                resource_id = str(incident_data["resourceId"])
                if "prod-" in resource_id or "/prod/" in resource_id:
                    env = "prod"
                elif "uat-" in resource_id or "/uat/" in resource_id:
                    env = "uat"
                elif "dev-" in resource_id or "/dev/" in resource_id:
                    env = "dev"
            
            if env in ["prod", "production"]:
                return 1.2, "prod"  # Increase severity for production
            elif env in ["uat", "staging", "test"]:
                return 1.0, "uat"   # Normal severity for UAT
            elif env in ["dev", "development"]:
                return 0.8, "dev"   # Reduce severity for development
        
        return 1.0, "unknown"  # Default factor
    
    def analyze_text(self, text, incident_data):
        """Analyze text using both rule-based and BERT-based approaches"""
        text = text.lower()
        text_embedding = self.model.encode(text)
        
        matches = []
        for pattern, info in self.pattern_info.items():
            similarity = 1 - cosine(text_embedding, info['embedding'])
            
            # Store all matches above threshold
            if similarity >= self.similarity_threshold:
                matches.append({
                    'pattern': pattern,
                    'similarity': similarity,
                    'severity_level': info['severity_level'],
                    'base_score': info['base_score'],
                    'category': info['category']
                })
        
        if not matches:
            return {
                "severity_level": "S4",
                "bert_score": 0.0,
                "rule_score": 10.0,
                "combined_score": 10.0,
                "matched_pattern": None,
                "matched_keywords": [],
                "all_matches": []
            }
        
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = matches[0]
        
        # Environment-based adjustment
        env_factor, env_name = self.get_environment_factor(incident_data)
        
        bert_score = best_match['similarity']
        rule_score = float(best_match['base_score'])
        
        # Apply environment factor to rule score
        adjusted_rule_score = rule_score * env_factor
        
        if self.use_ml:
            # Prepare features for ML model
            features = np.array([[
                bert_score,
                adjusted_rule_score,
                env_factor
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ML prediction
            severity_class = self.classifier.predict(features_scaled)[0]
            
            # Calculate combined score using learned weights
            combined_score = (
                self.weights['bert_weight'] * bert_score +
                self.weights['rule_weight'] * adjusted_rule_score +
                self.weights['env_weight'] * env_factor +
                self.weights['intercept']
            )
            
            # Map numeric prediction back to severity level
            severity_map = {0: 'S1', 1: 'S2', 2: 'S3', 3: 'S4'}
            final_severity = severity_map[severity_class]
        else:
            # Fallback to rule-based scoring
            bert_weight = 0.4
            rule_weight = 0.6
            
            # Calculate combined score (0-100 scale)
            combined_score = (
                (bert_score * 100 * bert_weight) +  # Scale BERT score to 0-100
                (adjusted_rule_score * rule_weight)
            )
        
        # Extract matched keywords from text
        matched_keywords = [
            keyword for keyword in text.split()
            if any(pattern.lower() in keyword for pattern, _ in self.pattern_info.items())
        ]
        
        # Determine severity level based on combined score
        if combined_score >= 80:
            final_severity = "S1"
        elif combined_score >= 60:
            final_severity = "S2"
        elif combined_score >= 40:
            final_severity = "S3"
        else:
            final_severity = "S4"
        
        return {
            "severity_level": final_severity,
            "bert_score": bert_score,
            "rule_score": adjusted_rule_score,
            "combined_score": combined_score,
            "matched_pattern": best_match['pattern'],
            "matched_keywords": matched_keywords,
            "all_matches": matches[:3]  # Top 3 matches for reference
        }

    def extract_analysis_text(self, incident_data):
        """Extract relevant text for analysis from incident data"""
        analysis_text = []
        
        if isinstance(incident_data, dict):
            # Extract error message
            if "properties" in incident_data and isinstance(incident_data["properties"], dict):
                if "error" in incident_data["properties"]:
                    error = incident_data["properties"]["error"]
                    if isinstance(error, dict):
                        analysis_text.append(str(error.get("message", "")))
            
            # Add operation name
            if "operationName" in incident_data:
                op_name = incident_data["operationName"]
                if isinstance(op_name, dict):
                    analysis_text.append(str(op_name.get("value", "")))
                else:
                    analysis_text.append(str(op_name))
            
            # Add status information
            if "status" in incident_data:
                status = incident_data["status"]
                if isinstance(status, dict):
                    analysis_text.append(str(status.get("value", "")))
                else:
                    analysis_text.append(str(status))
                    
            # Add category if available
            if "category" in incident_data:
                analysis_text.append(str(incident_data["category"]))

        return " ".join(filter(None, analysis_text))

    def map_incidents(self):
        """Map severities for all incidents in the database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Update table schema to include new columns
        cursor.execute("""
        DROP TABLE IF EXISTS enhanced_severity_mappings
        """)
        
        cursor.execute("""
        CREATE TABLE enhanced_severity_mappings (
            incident_id TEXT PRIMARY KEY,
            severity_level TEXT NOT NULL,
            bert_score REAL NOT NULL,
            rule_score REAL NOT NULL,
            combined_score REAL NOT NULL,
            matched_pattern TEXT,
            matched_keywords TEXT,
            top_matches TEXT,
            environment TEXT,
            source_json TEXT,
            FOREIGN KEY (incident_id) REFERENCES incident_logs(id)
        )
        """)

        # Get all incidents
        cursor.execute("SELECT id, incident_json FROM incident_logs")
        incidents = cursor.fetchall()
        print(f"Processing {len(incidents)} incidents...")

        for incident_id, incident_json in incidents:
            incident_data = json.loads(incident_json)
            analysis_text = self.extract_analysis_text(incident_data)
            
            # Analyze the text with incident context
            severity_info = self.analyze_text(analysis_text.strip(), incident_data)
            
            # Get environment
            environment = incident_data.get("properties", {}).get("environment", "unknown")
            
            # Insert the enhanced severity mapping
            cursor.execute("""
            INSERT INTO enhanced_severity_mappings 
            (incident_id, severity_level, bert_score, rule_score, 
             combined_score, matched_pattern, matched_keywords, 
                top_matches, environment, source_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_id,
                severity_info["severity_level"],
                float(severity_info["bert_score"]),
                float(severity_info["rule_score"]),
                float(severity_info["combined_score"]),
                severity_info["matched_pattern"],
                json.dumps(severity_info["matched_keywords"]),
                json.dumps([{
                    'pattern': m['pattern'],
                    'similarity': float(m['similarity']),
                    'severity_level': m['severity_level']
                } for m in severity_info["all_matches"]]),
                environment,
                incident_json
            ))

            if cursor.rowcount % 100 == 0:
                conn.commit()
                print(f"Processed {cursor.rowcount} incidents...")

        conn.commit()
        
        # Print summary
        cursor.execute("""
        SELECT 
            severity_level,
            COUNT(*) as count,
            AVG(bert_score) as avg_bert,
            AVG(rule_score) as avg_rule,
            AVG(combined_score) as avg_combined,
            COUNT(DISTINCT matched_pattern) as unique_patterns
        FROM enhanced_severity_mappings
        GROUP BY severity_level
        ORDER BY avg_combined DESC
        """)
        
        print("\nSeverity Distribution Summary:")
        print("Level | Count | Avg BERT | Avg Rule | Avg Combined | Patterns")
        print("-" * 65)
        for row in cursor.fetchall():
            print(f"{row[0]:<6} | {row[1]:>5} | {row[2]:>8.2f} | {row[3]:>8.2f} | "
                  f"{row[4]:>11.2f} | {row[5]:>8}")

        conn.close()

if __name__ == "__main__":
    mapper = EnhancedSeverityMapper()
    mapper.initialize_db()
    mapper.map_incidents()