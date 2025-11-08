import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

DB_PATH = "D:/incident_management/data/sqlite/incident_management_v2.db"

class EnhancedSeverityMapper:
    def __init__(self):
        # BERT model loading with fallback options
        try:
            self.model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
            print("Using distilbert-base model for semantic analysis")
        except Exception:
            try:
                self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                print("Using all-mpnet-base model for semantic analysis")
            except Exception:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("Using MiniLM model for semantic analysis")

        self.similarity_threshold = 0.65
        self.load_rules()

        # Load ML models if available
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
            self.weights = {
                'bert_weight': 0.35,
                'rule_weight': 0.45,
                'env_weight': 0.20
            }

    def load_rules(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pattern, severity_level, base_score, category, description 
            FROM severity_rules
        """)
        self.rules = cursor.fetchall()
        conn.close()

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

    def get_environment_factor(self, incident_data):
        if isinstance(incident_data, dict):
            env = None
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                env = props.get("environment", "").lower() or props.get("env", "").lower()
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

    def extract_analysis_text(self, incident_data):
        analysis_text = []
        if isinstance(incident_data, dict):
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                error = props.get("error", {})
                if isinstance(error, dict):
                    analysis_text.append(str(error.get("message", "")))
            op_name = incident_data.get("operationName", "")
            if isinstance(op_name, dict):
                analysis_text.append(str(op_name.get("value", "")))
            else:
                analysis_text.append(str(op_name))
            status = incident_data.get("status", "")
            if isinstance(status, dict):
                analysis_text.append(str(status.get("value", "")))
            else:
                analysis_text.append(str(status))
            category = incident_data.get("category", "")
            analysis_text.append(str(category))
        return " ".join(filter(None, analysis_text))

    def analyze_text(self, text, incident_data):
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
            final_severity = severity_map.get(severity_class, 'S4')
        else:
            bert_weight = 0.4
            rule_weight = 0.6
            combined_score = (
                (bert_score * 100 * bert_weight) + (adjusted_rule_score * rule_weight)
            )
            if combined_score >= 80:
                final_severity = "S1"
            elif combined_score >= 60:
                final_severity = "S2"
            elif combined_score >= 40:
                final_severity = "S3"
            else:
                final_severity = "S4"

        matched_keywords = [
            keyword for keyword in text.split()
            if any(pattern.lower() in keyword for pattern in self.pattern_info.keys())
        ]

        return {
            "severity_level": final_severity,
            "bert_score": bert_score,
            "rule_score": adjusted_rule_score,
            "combined_score": combined_score,
            "matched_pattern": best_match['pattern'],
            "matched_keywords": matched_keywords,
            "all_matches": matches[:3]
        }

    def process_unprocessed_incidents(self):
        """Process only incidents with processed_at IS NULL"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT payload_id, payload FROM incident_logs WHERE processed_at IS NULL
        """)
        incidents = cursor.fetchall()
        print(f"Processing {len(incidents)} incidents...")
        for incident_id, incident_json in incidents:
            try:
                incident_data = json.loads(incident_json)
            except Exception as e:
                print(f"Skipping incident {incident_id}: invalid JSON payload - {str(e)}")
                continue
            analysis_text = self.extract_analysis_text(incident_data)
            severity_info = self.analyze_text(analysis_text.strip(), incident_data)
            environment = incident_data.get("properties", {}).get("environment", "unknown")

                        
            cursor.execute("""
                INSERT OR REPLACE INTO enhanced_severity_mappings 
                (payload_id, severity_id, bert_score, rule_score, combined_score, matched_pattern, environment, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_id,
                severity_info["severity_level"],
                float(severity_info["bert_score"]),
                float(severity_info["rule_score"]),
                float(severity_info["combined_score"]),
                severity_info["matched_pattern"],
                environment,
                incident_json
            ))

            cursor.execute("UPDATE incident_logs SET processed_at = CURRENT_TIMESTAMP WHERE payload_id = ?", (incident_id,))
            if cursor.rowcount % 100 == 0:
                conn.commit()
                print(f"Processed {cursor.rowcount} incidents...")
        conn.commit()
        conn.close()
