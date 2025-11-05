import sqlite3
import json
import logging
import numpy as np
from scipy.spatial.distance import cosine
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)

DB_PATH = "C:\\Users\\GENAIKOLGPUSR73\\Desktop\\incident_management_local\\incident_management.db"
MODEL_PATH = "C:\\Users\\GENAIKOLGPUSR73\\Desktop\\bert-base-uncased"

class BertEmbedder:
    def __init__(self, model_path=MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        logging.info(f"Loaded BERT model from {model_path}")

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
        masked = last_hidden_state * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.squeeze().cpu().numpy()

class EnhancedSeverityMapper:
    def __init__(self):
        self.embedder = BertEmbedder()
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
            logging.info("ML models loaded")
        except Exception as e:
            logging.warning(f"ML models not found; fallback to rule-based approach: {e}")
            self.use_ml = False
            self.weights = {
                'bert_weight': 0.35,
                'rule_weight': 0.45,
                'env_weight': 0.20,
                'intercept': 0.0
            }

    def load_rules(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT pattern, severity_level, base_score, category, description FROM severity_rules")
        rules = cursor.fetchall()
        conn.close()

        self.pattern_info = {}
        for pattern, sev_level, base_score, category, description in rules:
            embedding = self.embedder.encode(pattern.lower())
            self.pattern_info[pattern] = {
                'embedding': embedding,
                'severity_level': sev_level,
                'base_score': base_score,
                'category': category,
                'description': description
            }

    def get_environment_factor(self, incident_data):
        env = None
        if isinstance(incident_data, dict):
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                env = props.get("environment") or props.get("env")
                if env:
                    env = env.lower()
            if not env:
                tags = incident_data.get("tags", {})
                if isinstance(tags, dict):
                    env = tags.get("Environment", "").lower()
            if not env:
                resource_id = str(incident_data.get("resourceId", "")).lower()
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
        text_embedding = self.embedder.encode(text.lower())

        matches = []
        for pattern, info in self.pattern_info.items():
            similarity = 1 - cosine(text_embedding, info["embedding"])
            if similarity >= self.similarity_threshold:
                matches.append({
                    "pattern": pattern,
                    "similarity": similarity,
                    "severity_level": info["severity_level"],
                    "base_score": info["base_score"],
                    "category": info["category"],
                })

        if not matches:
            return {
                "severity_level": "S4",
                "bert_score": 0.0,
                "rule_score": 10.0,
                "combined_score": 10.0,
                "matched_pattern": None,
                "matched_keywords": [],
                "all_matches": [],
            }

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        best_match = matches[0]

        env_factor, _ = self.get_environment_factor(incident_data)
        bert_score = best_match["similarity"]
        rule_score = float(best_match["base_score"])
        adjusted_rule_score = rule_score * env_factor

        if self.use_ml:
            features = np.array([[bert_score, adjusted_rule_score, env_factor]])
            features_scaled = self.scaler.transform(features)
            severity_class = self.classifier.predict(features_scaled)[0]
            combined_score = (
                self.weights["bert_weight"] * bert_score
                + self.weights["rule_weight"] * adjusted_rule_score
                + self.weights["env_weight"] * env_factor
                + self.weights.get("intercept", 0)
            )
            severity_map = {0: "S1", 1: "S2", 2: "S3", 3: "S4"}
            final_severity = severity_map.get(severity_class, "S4")
        else:
            combined_score = bert_score * 100 * 0.4 + adjusted_rule_score * 0.6
            if combined_score >= 80:
                final_severity = "S1"
            elif combined_score >= 60:
                final_severity = "S2"
            elif combined_score >= 40:
                final_severity = "S3"
            else:
                final_severity = "S4"

        matched_keywords = [
            k for k in text.split() if any(pattern.lower() in k for pattern in self.pattern_info)
        ]

        return {
            "severity_level": final_severity,
            "bert_score": bert_score,
            "rule_score": adjusted_rule_score,
            "combined_score": combined_score,
            "matched_pattern": best_match["pattern"],
            "matched_keywords": matched_keywords,
            "all_matches": matches[:3],
        }

    def extract_analysis_text(self, incident_data):
        parts = []
        if isinstance(incident_data, dict):
            props = incident_data.get("properties", {})
            if isinstance(props, dict):
                error = props.get("error")
                if isinstance(error, dict):
                    parts.append(str(error.get("message", "")))
            op = incident_data.get("operationName")
            if op:
                if isinstance(op, dict):
                    parts.append(str(op.get("value", "")))
                else:
                    parts.append(str(op))
            status = incident_data.get("status")
            if status:
                if isinstance(status, dict):
                    parts.append(str(status.get("value", "")))
                else:
                    parts.append(str(status))
            category = incident_data.get("category")
            if category:
                parts.append(str(category))
        return " ".join(filter(None, parts))

    def initialize_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS enhanced_severity_mappings (
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
            """
        )
        conn.commit()
        conn.close()

    def map_incidents(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS enhanced_severity_mappings")
        self.initialize_db()

        cursor.execute("SELECT id, incident_json FROM incident_logs")
        incidents = cursor.fetchall()
        logging.info(f"Processing {len(incidents)} incidents...")

        for incident_id, incident_json in incidents:
            incident_data = json.loads(incident_json)
            analysis_text = self.extract_analysis_text(incident_data)
            severity_info = self.analyze_text(analysis_text.strip(), incident_data)
            environment = incident_data.get("properties", {}).get("environment", "unknown")

            cursor.execute(
                """
                INSERT INTO enhanced_severity_mappings (
                    incident_id, severity_level, bert_score, rule_score, combined_score,
                    matched_pattern, matched_keywords, top_matches, environment,
                    source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    severity_info["severity_level"],
                    float(severity_info["bert_score"]),
                    float(severity_info["rule_score"]),
                    float(severity_info["combined_score"]),
                    severity_info["matched_pattern"],
                    json.dumps(severity_info["matched_keywords"]),
                    json.dumps(
                        [
                            {
                                "pattern": m["pattern"],
                                "similarity": float(m["similarity"]),
                                "severity_level": m["severity_level"],
                            }
                            for m in severity_info["all_matches"]
                        ]
                    ),
                    environment,
                    incident_json,
                ),
            )

            if cursor.rowcount % 100 == 0:
                conn.commit()
                logging.info(f"Processed {cursor.rowcount} incidents...")

        conn.commit()
        conn.close()


if __name__ == "__main__":
    mapper = EnhancedSeverityMapper()
    mapper.map_incidents()
