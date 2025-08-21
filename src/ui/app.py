"""
Streamlit web interface for Legal Document Review AI Agent.
"""

import streamlit as st
import requests
import time
import io
from pathlib import Path
import json
from typing import Optional, Dict, Any

from src.core.models import DocumentType, RiskLevel
from src.core.config import get_config

# Page configuration
st.set_page_config(
    page_title="Legal Document Review AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
config = get_config()

# API base URL (adjust if running on different host/port)
API_BASE_URL = f"http://{config.api.host}:{config.api.port}"

def main():
    """Main Streamlit application."""
    
    # Sidebar
    with st.sidebar:
        st.title("⚖️ Legal AI Assistant")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Choose a function:",
            [
                "📄 Document Review",
                "⚡ Quick Risk Assessment", 
                "📊 Dashboard",
                "⚙️ Settings"
            ]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "AI-powered legal document analysis with expert-level "
            "review capabilities including risk assessment, "
            "clause analysis, and redlining."
        )
    
    # Main content based on selected page
    if page == "📄 Document Review":
        document_review_page()
    elif page == "⚡ Quick Risk Assessment":
        quick_risk_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "⚙️ Settings":
        settings_page()


def document_review_page():
    """Main document review interface."""
    
    st.title("📄 Legal Document Review")
    st.markdown("Upload a legal document for comprehensive AI-powered analysis.")
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a legal document",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Supported formats: PDF, Word documents, Text files"
        )
    
    with col2:
        document_type = st.selectbox(
            "Document Type (Optional)",
            options=[None] + [dt.value for dt in DocumentType],
            format_func=lambda x: "Auto-detect" if x is None else x.replace("_", " ").title()
        )
        
        priority_areas = st.multiselect(
            "Priority Areas",
            options=[
                "Financial", "Operational", "Legal", 
                "Commercial", "Compliance", "IP", 
                "Data Privacy", "Termination"
            ],
            help="Focus analysis on specific risk areas"
        )
    
    # Analysis options
    with st.expander("Advanced Options"):
        custom_instructions = st.text_area(
            "Custom Instructions",
            placeholder="Specific areas to focus on or client requirements...",
            help="Provide specific instructions for the AI analysis"
        )
    
    # Start analysis
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Start Full Analysis", type="primary"):
                run_full_analysis(uploaded_file, document_type, priority_areas, custom_instructions)
        
        with col2:
            if st.button("⚡ Quick Risk Check"):
                run_quick_analysis(uploaded_file)
        
        # Display file info
        with col3:
            st.info(
                f"**File:** {uploaded_file.name}  \n"
                f"**Size:** {uploaded_file.size / 1024:.1f} KB  \n"
                f"**Type:** {uploaded_file.type}"
            )


def run_full_analysis(uploaded_file, document_type, priority_areas, custom_instructions):
    """Run full document analysis."""
    
    with st.spinner("Uploading document..."):
        # Upload file
        files = {"file": uploaded_file.getvalue()}
        upload_response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if upload_response.status_code != 200:
            st.error(f"Upload failed: {upload_response.text}")
            return
        
        file_data = upload_response.json()
        file_id = file_data["file_id"]
    
    # Start analysis
    analysis_request = {
        "file_id": file_id,
        "document_type": document_type,
        "priority_areas": priority_areas,
        "custom_instructions": custom_instructions
    }
    
    analysis_response = requests.post(
        f"{API_BASE_URL}/analyze", 
        json=analysis_request
    )
    
    if analysis_response.status_code != 200:
        st.error(f"Analysis failed: {analysis_response.text}")
        return
    
    workflow_data = analysis_response.json()
    workflow_id = workflow_data["workflow_id"]
    
    # Show progress
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Poll for results
    max_wait_time = 300  # 5 minutes
    poll_interval = 2    # 2 seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        status_response = requests.get(f"{API_BASE_URL}/status/{workflow_id}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            progress_placeholder.progress(status_data["progress"] / 100)
            status_placeholder.text(f"Status: {status_data['current_step']}")
            
            if status_data["status"] == "completed":
                # Get results
                result_response = requests.get(f"{API_BASE_URL}/result/{workflow_id}")
                if result_response.status_code == 200:
                    display_analysis_results(result_response.json())
                break
            elif status_data["status"] == "failed":
                st.error("Analysis failed. Please try again.")
                break
        
        time.sleep(poll_interval)
        elapsed_time += poll_interval
    else:
        st.error("Analysis timed out. Please try again.")


def run_quick_analysis(uploaded_file):
    """Run quick risk assessment."""
    
    with st.spinner("Performing quick risk assessment..."):
        # Upload file
        files = {"file": uploaded_file.getvalue()}
        upload_response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if upload_response.status_code != 200:
            st.error(f"Upload failed: {upload_response.text}")
            return
        
        file_data = upload_response.json()
        file_id = file_data["file_id"]
        
        # Quick assessment
        assessment_request = {"file_id": file_id}
        assessment_response = requests.post(
            f"{API_BASE_URL}/quick-risk-assessment",
            json=assessment_request
        )
        
        if assessment_response.status_code == 200:
            risk_data = assessment_response.json()
            display_risk_assessment(risk_data)
        else:
            st.error(f"Quick assessment failed: {assessment_response.text}")


def display_analysis_results(results: Dict[str, Any]):
    """Display comprehensive analysis results."""
    
    st.success("✅ Analysis completed successfully!")
    
    # Overview metrics
    st.markdown("## 📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = results["risk_assessment"]["overall_score"]
        score_color = get_risk_color(overall_score)
        st.metric(
            "Overall Risk Score",
            f"{overall_score}/100",
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        critical_issues = len(results["risk_assessment"]["critical_issues"])
        st.metric("Critical Issues", critical_issues)
    
    with col3:
        total_changes = len(results["redlined_document"]["changes"])
        st.metric("Suggested Changes", total_changes)
    
    with col4:
        confidence = int(results["confidence_score"] * 100)
        st.metric("Confidence", f"{confidence}%")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Summary",
        "⚠️ Risk Assessment", 
        "📝 Clause Analysis",
        "✏️ Redlined Changes",
        "📚 Legal Precedents"
    ])
    
    with tab1:
        display_executive_summary(results["executive_summary"])
    
    with tab2:
        display_risk_assessment(results["risk_assessment"])
    
    with tab3:
        display_clause_analysis(results["clause_analyses"])
    
    with tab4:
        display_redlined_changes(results["redlined_document"])
    
    with tab5:
        display_legal_precedents(results["legal_precedents"])


def display_executive_summary(summary: Dict[str, Any]):
    """Display executive summary."""
    
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Overall Assessment:** {summary['overall_assessment']}")
        st.markdown(f"**Business Impact:** {summary['business_impact']}")
        st.markdown(f"**Recommendation:** {summary['recommendation']}")
        st.markdown(f"**Estimated Review Time:** {summary['estimated_review_time']}")
    
    with col2:
        st.markdown("**Document Type:**")
        st.info(summary["document_type"].replace("_", " ").title())
    
    if summary.get("key_risks"):
        st.markdown("### 🚨 Key Risks")
        for risk in summary["key_risks"]:
            st.markdown(f"• {risk}")
    
    if summary.get("critical_action_items"):
        st.markdown("### ✅ Critical Action Items")
        for item in summary["critical_action_items"]:
            st.markdown(f"• {item}")


def display_risk_assessment(risk_data: Dict[str, Any]):
    """Display risk assessment results."""
    
    st.markdown("### ⚠️ Risk Assessment")
    
    # Overall score visualization
    overall_score = risk_data["overall_score"]
    risk_level = get_risk_level_from_score(overall_score)
    color = get_risk_color(overall_score)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### Overall Risk: {overall_score}/100")
        st.markdown(f"**Risk Level:** {risk_level}")
        
        # Progress bar with color
        st.progress(overall_score / 100)
    
    with col2:
        # Category scores
        if "category_scores" in risk_data:
            st.markdown("**Risk by Category:**")
            for category, score in risk_data["category_scores"].items():
                category_name = category.replace("_", " ").title()
                st.markdown(f"• **{category_name}:** {score}/100")
    
    # Risk issues
    if risk_data.get("risk_issues"):
        st.markdown("### 🔍 Identified Risk Issues")
        
        for issue in risk_data["risk_issues"]:
            risk_color = get_risk_level_color(issue["risk_level"])
            
            with st.expander(f"{get_risk_emoji(issue['risk_level'])} {issue['title']}"):
                st.markdown(f"**Risk Level:** {issue['risk_level'].title()}")
                st.markdown(f"**Category:** {issue['risk_category'].replace('_', ' ').title()}")
                st.markdown(f"**Description:** {issue['description']}")
                st.markdown(f"**Impact:** {issue['impact_description']}")
                st.markdown(f"**Mitigation:** {issue['mitigation_strategy']}")
                
                if issue.get("confidence_score"):
                    st.markdown(f"**Confidence:** {int(issue['confidence_score'] * 100)}%")
    
    # Recommendations
    if risk_data.get("recommendations"):
        st.markdown("### 💡 Recommendations")
        for rec in risk_data["recommendations"]:
            st.markdown(f"• {rec}")


def display_clause_analysis(clause_analyses: list):
    """Display clause analysis results."""
    
    st.markdown("### 📝 Clause Analysis")
    
    if not clause_analyses:
        st.info("No clause analyses available.")
        return
    
    # Sort by enforceability score (lowest first)
    sorted_analyses = sorted(
        clause_analyses, 
        key=lambda x: x.get("enforceability_score", 100)
    )
    
    for analysis in sorted_analyses[:10]:  # Show top 10
        enforceability = analysis.get("enforceability_score", 0)
        favorability = analysis.get("business_favorability", 0)
        
        # Color coding based on scores
        if enforceability < 60 or favorability < 50:
            status_color = "🔴"
        elif enforceability < 80 or favorability < 70:
            status_color = "🟡"
        else:
            status_color = "🟢"
        
        with st.expander(f"{status_color} Clause Analysis (E:{enforceability}% F:{favorability}%)"):
            st.markdown(f"**Clause Content:** {analysis['clause_content'][:200]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Enforceability", f"{enforceability}%")
            with col2:
                st.metric("Business Favorability", f"{favorability}%")
            
            if analysis.get("market_standard_comparison"):
                st.markdown(f"**Market Comparison:** {analysis['market_standard_comparison']}")
            
            if analysis.get("suggested_improvements"):
                st.markdown("**Suggested Improvements:**")
                for improvement in analysis["suggested_improvements"]:
                    st.markdown(f"• {improvement}")
            
            if analysis.get("legal_reasoning"):
                st.markdown(f"**Legal Reasoning:** {analysis['legal_reasoning']}")


def display_redlined_changes(redlined_doc: Dict[str, Any]):
    """Display redlined changes."""
    
    st.markdown("### ✏️ Redlined Changes")
    
    changes = redlined_doc.get("changes", [])
    
    if not changes:
        st.info("No changes suggested.")
        return
    
    # Group changes by type
    change_types = {}
    for change in changes:
        change_type = change["change_type"]
        if change_type not in change_types:
            change_types[change_type] = []
        change_types[change_type].append(change)
    
    # Display changes by type
    for change_type, type_changes in change_types.items():
        st.markdown(f"#### {change_type.replace('_', ' ').title()} ({len(type_changes)})")
        
        for change in type_changes:
            risk_emoji = get_risk_emoji(change["risk_level"])
            
            with st.expander(f"{risk_emoji} {change['comment'][:100]}..."):
                st.markdown(f"**Change Type:** {change['change_type'].replace('_', ' ').title()}")
                st.markdown(f"**Risk Level:** {change['risk_level'].title()}")
                
                if change["original_text"]:
                    st.markdown(f"**Original Text:** {change['original_text']}")
                
                if change["suggested_text"]:
                    st.markdown(f"**Suggested Text:** {change['suggested_text']}")
                
                st.markdown(f"**Comment:** {change['comment']}")
                st.markdown(f"**Reasoning:** {change['reasoning']}")
    
    # Summary comments
    if redlined_doc.get("comments"):
        st.markdown("### 💬 Summary Comments")
        for comment in redlined_doc["comments"]:
            st.info(comment)


def display_legal_precedents(precedents: list):
    """Display legal precedents."""
    
    st.markdown("### 📚 Legal Precedents")
    
    if not precedents:
        st.info("No legal precedents found.")
        return
    
    for precedent in precedents:
        confidence = int(precedent.get("confidence_score", 0) * 100)
        
        with st.expander(f"📖 {precedent['case_name']} (Confidence: {confidence}%)"):
            st.markdown(f"**Citation:** {precedent['citation']}")
            st.markdown(f"**Jurisdiction:** {precedent['jurisdiction']}")
            st.markdown(f"**Year:** {precedent['year']}")
            st.markdown(f"**Relevant Principle:** {precedent['relevant_principle']}")
            st.markdown(f"**Application:** {precedent['application_to_clause']}")


def quick_risk_page():
    """Quick risk assessment page."""
    
    st.title("⚡ Quick Risk Assessment")
    st.markdown("Get a rapid risk overview of your legal document.")
    
    uploaded_file = st.file_uploader(
        "Choose a document for quick risk check",
        type=['pdf', 'docx', 'doc', 'txt']
    )
    
    if uploaded_file is not None:
        if st.button("⚡ Analyze Risk", type="primary"):
            run_quick_analysis(uploaded_file)


def dashboard_page():
    """Dashboard with recent analyses."""
    
    st.title("📊 Dashboard")
    st.markdown("Overview of recent document analyses.")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("API Status", health_data["status"].title())
            
            with col2:
                st.metric("Active Workflows", health_data["active_workflows"])
            
            with col3:
                st.metric("Completed Reviews", health_data["completed_reviews"])
            
            # LLM provider status
            st.markdown("### 🤖 AI Provider Status")
            llm_health = health_data.get("llm_providers", {})
            
            col1, col2 = st.columns(2)
            with col1:
                openai_status = "✅ Connected" if llm_health.get("openai") else "❌ Disconnected"
                st.markdown(f"**OpenAI:** {openai_status}")
            
            with col2:
                anthropic_status = "✅ Connected" if llm_health.get("anthropic") else "❌ Disconnected"
                st.markdown(f"**Anthropic:** {anthropic_status}")
        
        else:
            st.error("Cannot connect to API server")
    
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")


def settings_page():
    """Settings and configuration page."""
    
    st.title("⚙️ Settings")
    st.markdown("Configure the Legal Document Review AI Agent.")
    
    # API Configuration
    st.markdown("### 🔗 API Configuration")
    current_api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    # Model Preferences
    st.markdown("### 🤖 AI Model Preferences")
    
    default_model = st.selectbox(
        "Default Language Model",
        options=["claude-3-sonnet", "gpt-4", "gpt-3.5-turbo", "claude-3-haiku"],
        index=0
    )
    
    temperature = st.slider(
        "Model Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower values are more conservative, higher values are more creative"
    )
    
    # Analysis Preferences
    st.markdown("### 📋 Analysis Preferences")
    
    risk_thresholds = st.columns(3)
    
    with risk_thresholds[0]:
        critical_threshold = st.number_input(
            "Critical Risk Threshold",
            min_value=50,
            max_value=100,
            value=80
        )
    
    with risk_thresholds[1]:
        high_threshold = st.number_input(
            "High Risk Threshold", 
            min_value=40,
            max_value=90,
            value=60
        )
    
    with risk_thresholds[2]:
        medium_threshold = st.number_input(
            "Medium Risk Threshold",
            min_value=20,
            max_value=80,
            value=40
        )
    
    # Save settings button
    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Note: This is a demo - settings are not persisted)")


# Utility functions
def get_risk_color(score: int) -> str:
    """Get color based on risk score."""
    if score >= 80:
        return "red"
    elif score >= 60:
        return "orange"
    elif score >= 40:
        return "yellow"
    else:
        return "green"


def get_risk_level_from_score(score: int) -> str:
    """Get risk level string from score."""
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def get_risk_level_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "critical": "red",
        "high": "orange", 
        "medium": "yellow",
        "low": "green",
        "minimal": "green"
    }
    return colors.get(risk_level.lower(), "gray")


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level."""
    emojis = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "⚡",
        "low": "ℹ️",
        "minimal": "✅"
    }
    return emojis.get(risk_level.lower(), "📋")


if __name__ == "__main__":
    main()
