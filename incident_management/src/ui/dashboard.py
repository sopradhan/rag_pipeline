import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

class Dashboard:
    def __init__(self):
        self.token = None
    
    def login_page(self):
        """Show login page and handle authentication."""
        st.title("🚨 Incident Management System")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                response = requests.post(
                    f"{API_URL}/token",
                    data={"username": username, "password": password}
                )
                
                if response.status_code == 200:
                    self.token = response.json()["access_token"]
                    st.session_state["token"] = self.token
                    st.success("Login successful!")
                else:
                    st.error("Invalid credentials")

    def incident_overview(self):
        """Display incident overview and metrics."""
        st.subheader("📊 Incident Overview")
        
        # Fetch incident data
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{API_URL}/incidents", headers=headers)
        incidents = response.json()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = len(incidents)
            st.metric("Total Incidents", total)
            
        with col2:
            active = sum(1 for i in incidents if i["status"] == "active")
            st.metric("Active Incidents", active)
            
        with col3:
            high_severity = sum(1 for i in incidents if i["severity"] == "HIGH")
            st.metric("High Severity", high_severity)
            
        with col4:
            mttr = self.calculate_mttr(incidents)
            st.metric("Avg. MTTR", f"{mttr:.2f}h")
        
        # Severity distribution chart
        severity_counts = pd.DataFrame([
            {"severity": i["severity"], "count": 1}
            for i in incidents
        ]).groupby("severity").sum()
        
        fig = px.pie(
            severity_counts,
            values="count",
            names=severity_counts.index,
            title="Incident Severity Distribution"
        )
        st.plotly_chart(fig)
        
        # Timeline chart
        timeline_df = pd.DataFrame([
            {
                "timestamp": i["created_at"],
                "severity": i["severity"]
            }
            for i in incidents
        ])
        timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
        
        fig = px.line(
            timeline_df.groupby([
                timeline_df["timestamp"].dt.date,
                "severity"
            ]).size().reset_index(),
            x="timestamp",
            y=0,
            color="severity",
            title="Incident Timeline"
        )
        st.plotly_chart(fig)

    def agent_performance(self):
        """Display agent performance metrics."""
        st.subheader("🤖 Agent Performance")
        
        # Fetch task data
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{API_URL}/tasks/status", headers=headers)
        tasks = response.json()
        
        # Agent success rates
        agent_stats = {}
        for task in tasks:
            agent = task["agent_type"]
            success = task["status"] == "completed"
            
            if agent not in agent_stats:
                agent_stats[agent] = {"total": 0, "success": 0}
            
            agent_stats[agent]["total"] += 1
            if success:
                agent_stats[agent]["success"] += 1
        
        # Create success rate chart
        success_rates = {
            agent: stats["success"] / stats["total"] * 100
            for agent, stats in agent_stats.items()
        }
        
        fig = px.bar(
            x=list(success_rates.keys()),
            y=list(success_rates.values()),
            title="Agent Success Rates (%)"
        )
        st.plotly_chart(fig)
        
        # Task processing times
        processing_times = pd.DataFrame([
            {
                "agent": t["agent_type"],
                "duration": (
                    datetime.fromisoformat(t["completed_at"]) -
                    datetime.fromisoformat(t["created_at"])
                ).total_seconds() / 60
            }
            for t in tasks if t["status"] == "completed"
        ])
        
        fig = px.box(
            processing_times,
            x="agent",
            y="duration",
            title="Task Processing Times (minutes)"
        )
        st.plotly_chart(fig)

    def incident_details(self):
        """Show detailed incident information and actions."""
        st.subheader("🔍 Incident Details")
        
        # Fetch incidents
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{API_URL}/incidents", headers=headers)
        incidents = response.json()
        
        # Incident selection
        selected_incident = st.selectbox(
            "Select Incident",
            options=[i["id"] for i in incidents],
            format_func=lambda x: f"Incident {x}"
        )
        
        if selected_incident:
            incident = next(i for i in incidents if i["id"] == selected_incident)
            
            # Display incident details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Status:**", incident["status"])
                st.write("**Severity:**", incident["severity"])
                st.write("**Created:**", incident["created_at"])
                
            with col2:
                st.write("**Type:**", incident["type"])
                st.write("**Department:**", incident["department"])
                st.write("**Assigned To:**", incident["assigned_to"])
            
            # Show incident description
            st.text_area(
                "Description",
                value=incident["description"],
                height=100,
                disabled=True
            )
            
            # Show AI analysis
            if "ai_analysis" in incident:
                st.subheader("🧠 AI Analysis")
                st.write(incident["ai_analysis"])
            
            # Actions
            st.subheader("Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Run Analysis"):
                    self.run_analysis(selected_incident)
            
            with col2:
                if st.button("Generate Report"):
                    self.generate_report(selected_incident)
            
            with col3:
                if st.button("Close Incident"):
                    self.close_incident(selected_incident)

    @staticmethod
    def calculate_mttr(incidents: List[Dict[str, Any]]) -> float:
        """Calculate Mean Time To Resolution in hours."""
        resolution_times = []
        for incident in incidents:
            if incident["status"] == "resolved":
                created = datetime.fromisoformat(incident["created_at"])
                resolved = datetime.fromisoformat(incident["resolved_at"])
                resolution_times.append((resolved - created).total_seconds() / 3600)
        
        return sum(resolution_times) / len(resolution_times) if resolution_times else 0

def main():
    st.set_page_config(
        page_title="Incident Management Dashboard",
        page_icon="🚨",
        layout="wide"
    )
    
    dashboard = Dashboard()
    
    # Check authentication
    if "token" not in st.session_state:
        dashboard.login_page()
    else:
        dashboard.token = st.session_state["token"]
        
        # Sidebar navigation
        page = st.sidebar.radio(
            "Navigation",
            ["Overview", "Agent Performance", "Incident Details"]
        )
        
        if page == "Overview":
            dashboard.incident_overview()
        elif page == "Agent Performance":
            dashboard.agent_performance()
        else:
            dashboard.incident_details()

if __name__ == "__main__":
    main()