"""Metrics collection and monitoring for RAG pipeline."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import json
import numpy as np
from collections import defaultdict

class MetricsCollector:
    """System metrics collection and monitoring."""
    
    def __init__(self, metrics_file: str = "metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_cache = defaultdict(list)
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file."""
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics_cache.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_cache, f)
    
    def record_metric(self, metric_type: str, value: float):
        """Record a new metric value."""
        timestamp = datetime.now().isoformat()
        self.metrics_cache[metric_type].append({
            "timestamp": timestamp,
            "value": value
        })
        self._save_metrics()
    
    def get_total_documents(self) -> int:
        """Get total number of documents in the system."""
        return len(self.metrics_cache.get("documents", []))
    
    def get_cluster_count(self) -> int:
        """Get total number of clusters."""
        return len(self.metrics_cache.get("clusters", []))
    
    def get_avg_cluster_size(self) -> float:
        """Get average cluster size."""
        clusters = self.metrics_cache.get("clusters", [])
        if not clusters:
            return 0.0
        sizes = [c.get("size", 0) for c in clusters]
        return np.mean(sizes)
    
    def get_security_incidents_24h(self) -> int:
        """Get number of security incidents in last 24 hours."""
        now = datetime.now()
        incidents = self.metrics_cache.get("security_incidents", [])
        
        recent = [
            i for i in incidents
            if now - datetime.fromisoformat(i["timestamp"]) <= timedelta(hours=24)
        ]
        return len(recent)
    
    def get_activity_history(self, days: int = 7) -> List[Dict]:
        """Get system activity history."""
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        
        activities = []
        for activity_type in ["ingestion", "classification", "embedding", "search"]:
            type_activities = self.metrics_cache.get(f"{activity_type}_activity", [])
            recent = [
                {
                    "timestamp": a["timestamp"],
                    "count": a["count"],
                    "activity_type": activity_type
                }
                for a in type_activities
                if datetime.fromisoformat(a["timestamp"]) >= cutoff
            ]
            activities.extend(recent)
        
        return sorted(activities, key=lambda x: x["timestamp"])
    
    def get_classification_distribution(self) -> Dict[str, int]:
        """Get distribution of document classifications."""
        classifications = self.metrics_cache.get("classifications", [])
        dist = defaultdict(int)
        for c in classifications:
            dist[c["category"]] += 1
        return dict(dist)
    
    def get_guardrails_metrics(self) -> Dict:
        """Get content guardrails metrics."""
        metrics = {
            "pii_types": defaultdict(int),
            "actions": defaultdict(int)
        }
        
        guardrails = self.metrics_cache.get("guardrails", [])
        for g in guardrails:
            if "pii_type" in g:
                metrics["pii_types"][g["pii_type"]] += 1
            if "action" in g:
                metrics["actions"][g["action"]] += 1
        
        return {
            "pii_types": dict(metrics["pii_types"]),
            "actions": dict(metrics["actions"])
        }
    
    def get_performance_metrics(self, time_range: str) -> List[Dict]:
        """Get system performance metrics."""
        now = datetime.now()
        
        if time_range == "Last 24 Hours":
            cutoff = now - timedelta(hours=24)
        elif time_range == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        else:  # Last 30 Days
            cutoff = now - timedelta(days=30)
        
        perf_metrics = self.metrics_cache.get("performance", [])
        recent = [
            m for m in perf_metrics
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
        
        return recent
    
    def get_resource_metrics(self) -> Dict:
        """Get current system resource usage."""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "storage_usage": psutil.disk_usage('/').percent
        }
    
    def clear_old_metrics(self, days: int = 30):
        """Clear metrics older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        for metric_type in self.metrics_cache:
            self.metrics_cache[metric_type] = [
                m for m in self.metrics_cache[metric_type]
                if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]
        
        self._save_metrics()