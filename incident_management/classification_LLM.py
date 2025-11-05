import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from transformers import pipeline


# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"


class EnhancedSeverityMapper:
    def __init__(self):
        # Initialize sentence transformer model with fallback
        try:
            self.model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
            print("Using distilbert-base model for semantic analysis")
        except Exception as e1:
            try:
                self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                print("Using all-mpnet-base model for semantic analysis")
            except Exception as e2:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("Using MiniLM model for semantic analysis")

        self.load_rules()
        self.similarity_threshold = 0.65  # Threshold for semantic similarity

        # Load ML-based models if present
        try:
            import joblib
            self.scaler = joblib.load('incident_severity_scaler.joblib')
            self.classifier = joblib.load('incident_severity_classifier.joblib')
            with open('severity_weights.json', 'r') as f:
                self.weights = json.load(f)
            self.use_ml = True
            print("Using ML-based severity classification")
        except Exception as e:
            print(f"ML models not found, using enhanced rule-based classification: {e}")
            self.use_ml = False
            self.weights = {
                'bert_weight': 0.35,
                'rule_weight': 0.45,
                'env_weight': 0.20,
                'intercept': 0
            }

        # Initialize Hugging Face LLM pipeline for corrective action generation
        try:
            self.llm_generator = pipeline(
                "text-generation",
                model="bigscience/bloom-1b7",  # Replace with preferred HF model
                device=0  # -1 for CPU
            )
            print("Initialized Hugging Face LLM generator")
        except Exception as e:
            print(f"Error initializing Hugging Face LLM pipeline: {e}")
            self.llm_generator = None

    def load_rules(self):
        """Load severity rules from DB and precompute embeddings"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pattern, severity_level, base_score, category, description
            FROM severity_rules
        """)
        self.rules = cursor.fetchall()
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
        """Create enhanced_severity_mappings table with corrective_action column"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS enhanced_severity_mappings")
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
                corrective_action TEXT,
                source_json TEXT,
                FOREIGN KEY (incident_id) REFERENCES incident_logs(id)
            )
        """)
        conn.commit()
        conn.close()

    def get_environment_factor(self, incident_data):
        """Calculate environment factor based on incident data"""
        if isinstance(incident_data, dict):
            env = None
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                env = props.get("environment", "") or props.get("env", "")
                env = env.lower() if env else None

            if not env:
                tags = incident_data.get("tags", {})
                if isinstance(tags, dict):
                    env = tags.get("Environment", "").lower()

            if not env:
                resource_id = str(incident_data.get("resourceId", ""))
                if "prod-" in resource_id or "/prod/" in resource_id:
                    env = "prod"
                elif "uat-" in resource_id or "/uat/" in resource_id:
                    env = "uat"
                elif "dev-" in resource_id or "/dev/" in resource_id:
                    env = "dev"

            if env in ["prod", "production"]:
                return 1.2, "prod"
            elif env in ["uat", "staging", "test"]:
                return 1.0, "uat"
            elif env in ["dev", "development"]:
                return 0.8, "dev"
        return 1.0, "unknown"

    def analyze_text(self, text, incident_data):
        """Analyze incident text to determine severity and match patterns"""
        text = text.lower()
        text_embedding = self.model.encode(text)

        matches = []
        for pattern, info in self.pattern_info.items():
            similarity = 1 - cosine(text_embedding, info['embedding'])
            if similarity >= self.similarity_threshold:
                matches.append({
                    'pattern': pattern,
                    'similarity': similarity,
                    'severity_level': info['severity_level'],
                    'base_score': info['base_score'],
                    'category': info['category'],
                    'description': info.get('description', '')
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

        matches.sort(key=lambda x: x['similarity'], reverse=True)
        best_match = matches[0]

        env_factor, env_name = self.get_environment_factor(incident_data)

        bert_score = best_match['similarity']
        rule_score = float(best_match['base_score'])
        adjusted_rule_score = rule_score * env_factor

        if self.use_ml:
            features = np.array([[bert_score, adjusted_rule_score, env_factor]])
            features_scaled = self.scaler.transform(features)
            severity_class = self.classifier.predict(features_scaled)[0]
            combined_score = (
                self.weights['bert_weight'] * bert_score +
                self.weights['rule_weight'] * adjusted_rule_score +
                self.weights['env_weight'] * env_factor +
                self.weights.get('intercept', 0)
            )
            severity_map = {0: 'S1', 1: 'S2', 2: 'S3', 3: 'S4'}
            final_severity = severity_map.get(severity_class, "S4")
        else:
            bert_weight = 0.4
            rule_weight = 0.6
            combined_score = (bert_score * 100 * bert_weight) + (adjusted_rule_score * rule_weight)

            if combined_score >= 80:
                final_severity = "S1"
            elif combined_score >= 60:
                final_severity = "S2"
            elif combined_score >= 40:
                final_severity = "S3"
            else:
                final_severity = "S4"

        matched_keywords = [kw for kw in text.split() if any(p.lower() in kw for p in self.pattern_info.keys())]

        return {
            "severity_level": final_severity,
            "bert_score": bert_score,
            "rule_score": adjusted_rule_score,
            "combined_score": combined_score,
            "matched_pattern": best_match['pattern'],
            "matched_keywords": matched_keywords,
            "all_matches": matches[:3]
        }

    def extract_analysis_text(self, incident_data):
        """Extract relevant textual information from incident data"""
        analysis_text = []
        if isinstance(incident_data, dict):
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                error = props.get("error", {})
                if isinstance(error, dict):
                    analysis_text.append(str(error.get("message", "")))
            op_name = incident_data.get("operationName")
            if op_name:
                if isinstance(op_name, dict):
                    analysis_text.append(str(op_name.get("value", "")))
                else:
                    analysis_text.append(str(op_name))
            status = incident_data.get("status")
            if status:
                if isinstance(status, dict):
                    analysis_text.append(str(status.get("value", "")))
                else:
                    analysis_text.append(str(status))
            category = incident_data.get("category")
            if category:
                analysis_text.append(str(category))

        return " ".join(filter(None, analysis_text))

    def llm_agent(self, severity, incident_description, matched_pattern):
        """Generate corrective action using the Hugging Face LLM"""
        if not self.llm_generator:
            return "LLM not available for corrective action."

        prompt = f"""
Incident Details: {incident_description}
Matched Pattern: {matched_pattern}
Assigned Severity: {severity}

Please validate the severity and generate clear, step-by-step corrective actions for this incident.
"""

        try:
            outputs = self.llm_generator(prompt, max_length=200, do_sample=True, temperature=0.7)
            corrective_action = outputs[0]["generated_text"].strip()
            # Optionally remove input prompt from generated_text if model echoes it
            if corrective_action.startswith(prompt):
                corrective_action = corrective_action[len(prompt):].strip()
            return corrective_action
        except Exception as e:
            return f"Error during corrective action generation: {e}"

    def map_incidents(self):
        """Main method to map incidents and generate corrective actions"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS enhanced_severity_mappings")
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
                corrective_action TEXT,
                source_json TEXT,
                FOREIGN KEY (incident_id) REFERENCES incident_logs(id)
            )
        """)

        cursor.execute("SELECT id, incident_json FROM incident_logs")
        incidents = cursor.fetchall()
        print(f"Processing {len(incidents)} incidents...")

        for incident_id, incident_json in incidents:
            incident_data = json.loads(incident_json)
            analysis_text = self.extract_analysis_text(incident_data)

            severity_info = self.analyze_text(analysis_text.strip(), incident_data)

            environment = incident_data.get("properties", {}).get("environment", "unknown")

            corrective_action = self.llm_agent(
                severity_info["severity_level"],
                analysis_text.strip(),
                severity_info["matched_pattern"]
            )

            cursor.execute("""
                INSERT INTO enhanced_severity_mappings
                (incident_id, severity_level, bert_score, rule_score,
                 combined_score, matched_pattern, matched_keywords,
                 top_matches, environment, corrective_action, source_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                corrective_action,
                incident_json
            ))

            if cursor.rowcount % 100 == 0:
                conn.commit()
                print(f"Processed {cursor.rowcount} incidents...")

        conn.commit()

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
