"""Streamlit dashboard for RAG pipeline visualization and monitoring."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional

from ..security.manager import SecurityManager
from ..security.guardrails import ContentGuardrails, ContentType
from ..embed.manager import EmbeddingManager
from ..classify.hybrid import HybridClassifier
from ..monitoring.metrics import MetricsCollector
from ..monitoring.audit import AuditLogger

class Dashboard:
    """Main dashboard application."""
    
    def __init__(
        self,
        security_manager: SecurityManager,
        embedding_manager: EmbeddingManager,
        classifier: HybridClassifier,
        metrics_collector: MetricsCollector,
        audit_logger: AuditLogger
    ):
        self.security = security_manager
        self.embeddings = embedding_manager
        self.classifier = classifier
        self.metrics = metrics_collector
        self.audit = audit_logger
        
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(page_title="RAG Pipeline Dashboard", layout="wide")
        
        # Authentication
        if not self._check_auth():
            return
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Clusters", "Security", "Monitoring", "Audit Logs"]
        )
        
        # Page routing
        if page == "Overview":
            self._show_overview()
        elif page == "Clusters":
            self._show_clusters()
        elif page == "Security":
            self._show_security()
        elif page == "Monitoring":
            self._show_monitoring()
        else:
            self._show_audit_logs()
    
    def _check_auth(self) -> bool:
        """Handle user authentication."""
        if "user_id" not in st.session_state:
            st.title("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if self.security.authenticate(username, password):
                    st.session_state.user_id = username
                    return True
                st.error("Invalid credentials")
                return False
            return False
        return True
    
    def _show_overview(self):
        """Display overview dashboard."""
        st.title("RAG Pipeline Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = self.metrics.get_total_documents()
            st.metric("Total Documents", total_docs)
            
        with col2:
            cluster_count = self.metrics.get_cluster_count()
            st.metric("Total Clusters", cluster_count)
            
        with col3:
            avg_cluster_size = self.metrics.get_avg_cluster_size()
            st.metric("Avg Cluster Size", f"{avg_cluster_size:.2f}")
            
        with col4:
            security_incidents = self.metrics.get_security_incidents_24h()
            st.metric("Security Incidents (24h)", security_incidents)
        
        # Recent activity chart
        st.subheader("System Activity (Last 7 Days)")
        activity_data = self.metrics.get_activity_history(days=7)
        fig = px.line(activity_data, x="timestamp", y="count", 
                     color="activity_type", title="Activity Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification distribution
        st.subheader("Document Classifications")
        class_dist = self.metrics.get_classification_distribution()
        fig = px.pie(values=list(class_dist.values()), 
                    names=list(class_dist.keys()),
                    title="Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_clusters(self):
        """Display cluster visualization and analysis."""
        st.title("Document Clusters")
        
        # Cluster selection
        clusters = self.embeddings.get_clusters()
        selected_cluster = st.selectbox(
            "Select Cluster",
            options=[c.id for c in clusters]
        )
        
        # Cluster details
        cluster = self.embeddings.get_cluster(selected_cluster)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Statistics")
            st.write(f"Size: {len(cluster.documents)}")
            st.write(f"Average Similarity: {cluster.avg_similarity:.3f}")
            st.write(f"Created: {cluster.created_at}")
            
            # Top terms
            st.subheader("Top Terms")
            terms = self.classifier.get_cluster_terms(selected_cluster)
            for term, weight in terms[:10]:
                st.write(f"- {term}: {weight:.3f}")
        
        with col2:
            # Document scatter plot
            embeddings = self.embeddings.get_cluster_embeddings(selected_cluster)
            fig = self._plot_embeddings(embeddings, cluster.labels)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_security(self):
        """Display security monitoring and configuration."""
        st.title("Security Dashboard")
        
        # Access control status
        st.subheader("Access Control")
        rbac_status = self.security.get_rbac_status()
        st.json(rbac_status)
        
        # Content guardrails
        st.subheader("Content Guardrails")
        guardrails_metrics = self.metrics.get_guardrails_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PII detection stats
            fig = px.bar(
                x=list(guardrails_metrics["pii_types"].keys()),
                y=list(guardrails_metrics["pii_types"].values()),
                title="PII Detection Stats"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Content filtering actions
            fig = px.pie(
                values=list(guardrails_metrics["actions"].values()),
                names=list(guardrails_metrics["actions"].keys()),
                title="Content Filter Actions"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_monitoring(self):
        """Display system monitoring metrics."""
        st.title("System Monitoring")
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )
        
        # System metrics
        st.subheader("System Performance")
        perf_metrics = self.metrics.get_performance_metrics(time_range)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time graph
            fig = px.line(
                perf_metrics,
                x="timestamp",
                y="response_time",
                title="Average Response Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error rate graph
            fig = px.line(
                perf_metrics,
                x="timestamp",
                y="error_rate",
                title="Error Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Resource usage
        st.subheader("Resource Usage")
        resource_metrics = self.metrics.get_resource_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{resource_metrics['cpu_usage']:.1f}%")
        
        with col2:
            st.metric("Memory Usage", f"{resource_metrics['memory_usage']:.1f}%")
        
        with col3:
            st.metric("Storage Usage", f"{resource_metrics['storage_usage']:.1f}%")
    
    def _show_audit_logs(self):
        """Display audit logs with filtering and search."""
        st.title("Audit Logs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_type = st.multiselect(
                "Event Type",
                self.audit.get_event_types()
            )
        
        with col2:
            severity = st.multiselect(
                "Severity",
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            )
        
        with col3:
            user = st.multiselect(
                "User",
                self.audit.get_users()
            )
        
        # Search
        search = st.text_input("Search Logs")
        
        # Get filtered logs
        logs = self.audit.get_logs(
            event_type=event_type,
            severity=severity,
            user=user,
            search=search
        )
        
        # Display logs
        st.dataframe(
            pd.DataFrame(logs),
            use_container_width=True
        )
    
    def _plot_embeddings(self, embeddings: List[List[float]], labels: List[str]):
        """Create scatter plot of document embeddings."""
        # Use PCA or t-SNE to reduce dimensions for visualization
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels
        })
        
        fig = px.scatter(
            df, x="x", y="y",
            color="label",
            title="Document Embeddings"
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    import os
    from ..security.manager import SecurityManager
    from ..embed.manager import EmbeddingManager
    from ..classify.hybrid import HybridClassifier
    from ..monitoring.metrics import MetricsCollector
    from ..monitoring.audit import AuditLogger
    
    # Initialize components
    security = SecurityManager()
    embeddings = EmbeddingManager()
    classifier = HybridClassifier()
    metrics = MetricsCollector()
    audit = AuditLogger()
    
    # Create and run dashboard
    dashboard = Dashboard(
        security_manager=security,
        embedding_manager=embeddings,
        classifier=classifier,
        metrics_collector=metrics,
        audit_logger=audit
    )
    
    dashboard.run()