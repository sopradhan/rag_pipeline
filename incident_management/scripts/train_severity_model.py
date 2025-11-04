import sqlite3
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Database configuration
DB_PATH = "D:/incident_management/data/sqlite/incident_management.db"

class SeverityModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.severity_map = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3}
        self.reverse_severity_map = {v: k for k, v in self.severity_map.items()}
        
    def prepare_training_data(self):
        """Prepare training data from existing severity mappings"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get features and labels from enhanced severity mappings
        cursor.execute("""
        SELECT 
            bert_score,
            rule_score,
            CASE 
                WHEN environment = 'prod' THEN 1.2
                WHEN environment = 'uat' THEN 1.0
                WHEN environment = 'dev' THEN 0.8
                ELSE 1.0
            END as env_factor,
            severity_level
        FROM enhanced_severity_mappings
        WHERE severity_level IS NOT NULL
        """)
        
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            raise ValueError("No training data available")
        
        # Prepare features and labels
        X = np.array([[row[0], row[1], row[2]] for row in data])
        y = np.array([self.severity_map[row[3]] for row in data])
        
        return X, y
    
    def train_model(self):
        """Train the severity classification model"""
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Get unique classes
        unique_classes = np.unique(y_train)
        target_names = [self.reverse_severity_map[i] for i in sorted(unique_classes)]
        
        print("\nModel Performance:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=target_names
        ))
        
        # Save models
        joblib.dump(self.scaler, 'incident_severity_scaler.joblib')
        joblib.dump(self.classifier, 'incident_severity_classifier.joblib')
        
        # Get feature importance
        coefficients = self.classifier.coef_
        feature_names = ['BERT Score', 'Rule Score', 'Environment Factor']
        
        print("\nFeature Importance:")
        for i, feature in enumerate(feature_names):
            importance = np.abs(coefficients[:, i]).mean()
            print(f"{feature}: {importance:.4f}")
        
        return self.classifier.score(X_test_scaled, y_test)
    
    def optimize_weights(self):
        """Learn optimal weights for combining scores"""
        X, y = self.prepare_training_data()
        
        # Convert severity levels to numeric scores (S1=100, S2=75, S3=50, S4=25)
        severity_scores = {
            0: 100,  # S1
            1: 75,   # S2
            2: 50,   # S3
            3: 25    # S4
        }
        y_scores = np.array([severity_scores[label] for label in y])
        
        # Fit linear regression to learn weights
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, y_scores)
        
        # Get optimal weights
        weights = {
            'bert_weight': reg.coef_[0],
            'rule_weight': reg.coef_[1],
            'env_weight': reg.coef_[2],
            'intercept': reg.intercept_
        }
        
        # Save weights
        with open('severity_weights.json', 'w') as f:
            json.dump(weights, f, indent=2)
        
        print("\nOptimized Weights:")
        print(f"BERT Score Weight: {weights['bert_weight']:.4f}")
        print(f"Rule Score Weight: {weights['rule_weight']:.4f}")
        print(f"Environment Weight: {weights['env_weight']:.4f}")
        print(f"Intercept: {weights['intercept']:.4f}")
        
        return weights

def main():
    trainer = SeverityModelTrainer()
    
    print("Training severity classification model...")
    accuracy = trainer.train_model()
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    print("\nOptimizing score weights...")
    weights = trainer.optimize_weights()
    
    # Save results summary
    results = {
        'accuracy': accuracy,
        'weights': weights,
        'timestamp': '2025-11-04'  # Current date
    }
    
    with open('severity_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nModel and weights have been saved and can be used in the enhanced severity mapper.")

if __name__ == "__main__":
    main()