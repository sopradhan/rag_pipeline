incident_severity_classification/
├── data/
│   ├── raw/                      # Raw system logs and incident metrics files (CSV, JSON, etc.)
│   ├── processed/                # Preprocessed and tokenized data files (optional)
│   ├── labels.csv                # Labeled data mapping texts to severity classes (S1, S2, S3, S4)
│
├── models/
│   ├── bert-base-uncased/        # Your downloaded pretrained BERT uncased model files
│   ├── fine_tuned/               # Folder to save your fine-tuned BERT model and tokenizer
│   ├── incident_severity_scaler.joblib  # Optional scaler for features if using ML models
│   ├── incident_severity_classifier.joblib  # Optional classifier model saved with joblib
│   └── severity_weights.json     # Optional weight file used in classification
│
├── notebooks/                   # Jupyter notebooks for experimentation and EDA
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Scripts to load, clean, and preprocess data
│   ├── dataset.py              # Dataset classes for PyTorch or other frameworks
│   ├── train.py                # Training script for fine-tuning BERT
│   ├── evaluate.py             # Evaluation scripts
│   ├── enhanced_severity_mapper.py # Your main severity mapping class
│   └── utils.py                # Utilities like metrics, helpers, etc.
│
├── scripts/                     # Command-line script wrappers like app.py
│   └── app.py                  # Runs severity classification process (calls EnhancedSeverityMapper)
│
├── requirements.txt             # Python dependencies list (transformers, torch, scikit-learn, etc.)
├── README.md                   # Project overview, setup instructions
└── .gitignore                  # Ignore data, models, env files for git
