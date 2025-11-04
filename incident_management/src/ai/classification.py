from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from .models import IncidentSeverity

class BERTClassifier:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # LOW, MEDIUM, HIGH
        )
        self.model.eval()
        
    async def classify(self, text: str) -> Tuple[IncidentSeverity, float]:
        """
        Classify incident severity using BERT.
        Returns severity and confidence score.
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs).item()
            confidence = probs[0][prediction].item()
        
        # Map prediction to severity
        severity_map = {
            0: IncidentSeverity.LOW,
            1: IncidentSeverity.MEDIUM,
            2: IncidentSeverity.HIGH
        }
        
        return severity_map[prediction], confidence

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # Tokenize and prepare input
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(
                model_output,
                encoded_input['attention_mask']
            )
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings.tolist()

# Global instances
classifier = BERTClassifier()
embedding_generator = EmbeddingGenerator()

async def classify_severity(text: str) -> IncidentSeverity:
    """Wrapper function to classify incident severity."""
    severity, _ = await classifier.classify(text)
    return severity

async def generate_embeddings(text: str) -> List[float]:
    """Wrapper function to generate embeddings."""
    embeddings = await embedding_generator.generate([text])
    return embeddings[0]