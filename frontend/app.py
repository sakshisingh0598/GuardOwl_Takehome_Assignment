import streamlit as st
import requests
import json
from datetime import datetime, date
from typing import List, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Guard Owl AI",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class GuardOwlClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def query_reports(self, query: str, site_id: Optional[str] = None, 
                     date_range: Optional[List[str]] = None) -> dict:
        """Query reports via API"""
        payload = {"query": query}
        if site_id:
            payload["siteId"] = site_id
        if date_range:
            payload["dateRange"] = date_range
        
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def format_datetime(date_str: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return date_str

def main():
    # Initialize client
    client = GuardOwlClient(API_BASE_URL)
    
    # Header
    st.title("🦉 Guard Owl AI")
    st.markdown("Ask questions about security reports using natural language")
    
    # Sidebar for filters and status
    with st.sidebar:
        st.header("Filters & Settings")
        
        # API Health Status
        st.subheader("Service Status")
        health = client.health_check()
        
        if health.get("status") == "healthy":
            st.success("API Connected")
            st.metric("Reports Loaded", health.get("reports_loaded", 0))
        else:
            st.error("API Disconnected")
            st.error(health.get("message", "Unknown error"))
        
        st.divider()
        
        # Site Filter
        st.subheader("Filters")
        site_options = ["All Sites", "S01", "S02", "S03", "S04", "S05"]
        selected_site = st.selectbox("Site ID", site_options)
        site_filter = None if selected_site == "All Sites" else selected_site
        
        # Date Range Filter
        use_date_filter = st.checkbox("Filter by Date Range")
        date_filter = None
        
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=date(2025, 8, 25))
            with col2:
                end_date = st.date_input("End Date", value=date(2025, 9, 3))
            
            if start_date <= end_date:
                date_filter = [
                    start_date.strftime("%Y-%m-%dT00:00:00Z"),
                    end_date.strftime("%Y-%m-%dT23:59:59Z")
                ]
            else:
                st.error("Start date must be before end date")
        
        st.divider()
        
        # Example queries
        st.subheader("Example Queries")
        example_queries = [
            "What happened at Site S01 last night?",
            "Were there any geofence breaches?",
            "Show me all incidents involving a red Toyota Camry",
            "Any suspicious vehicle activity?",
            "Reports about the west gate",
            "Tailgating incidents"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{example[:10]}"):
                st.session_state.main_query = example
                st.rerun()
    
    # Main content area
    st.markdown("**Ask a question about security reports:**")
    
    # Use a form to enable Enter key submission
    with st.form(key="search_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Query input (no label since we have header above)
            query = st.text_input(
                label="query_input",
                placeholder="e.g., What happened at Site S01 last night?",
                key="main_query",
                label_visibility="collapsed"
            )
        
        with col2:
            # Button will now align with input field
            search_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
    
    # Process query
    if search_button and query:
        if health.get("status") != "healthy":
            st.error("Cannot process query: API service is not available")
            return
        
        with st.spinner("Searching reports..."):
            result = client.query_reports(query, site_filter, date_filter)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Display results
            st.subheader("📋 Results")
            
            # Summary
            st.markdown("### Summary")
            st.write(result.get("answer", "No answer provided"))
            
            # Reports details
            reports = result.get("reports", [])
            if reports:
                st.markdown(f"### Relevant Reports ({len(reports)} found)")
                
                for i, report in enumerate(reports, 1):
                    with st.expander(f"Report {report['id']} - {report['siteId']} ({format_datetime(report['date'])})"):
                        col_left, col_right = st.columns([3, 1])
                        
                        with col_left:
                            st.markdown("**Report Details:**")
                            st.write(report['text'])
                        
                        with col_right:
                            st.markdown("**Metadata:**")
                            st.text(f"ID: {report['id']}")
                            st.text(f"Site: {report['siteId']}")
                            st.text(f"Guard: {report['guardId']}")
                            st.text(f"Date: {format_datetime(report['date'])}")
            else:
                st.info("No relevant reports found for your query.")
            
            # Sources
            sources = result.get("sources", [])
            if sources:
                st.markdown("### Sources")
                st.code(", ".join(sources))
    
    elif search_button and not query:
        st.warning("Please enter a query to search")
    
    # Footer
    st.markdown("---")
    st.markdown("Guard Owl AI - Semantic Search for Security Reports")

if __name__ == "__main__":
    main()
