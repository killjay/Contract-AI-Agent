"""
Standalone Streamlit app for Legal Document Review AI Agent.
This version works without the API server.
"""

import streamlit as st
import tempfile
import asyncio
from pathlib import Path
import json
from typing import Optional
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.models import DocumentType, RiskLevel, AnalysisResult
    from src.core.config import get_config
    from src.parsers.document_parser import DocumentParser
    from src.analysis.legal_analyzer import LegalAnalyzer
    from src.analysis.risk_assessor import RiskAssessor
    from src.redlining.redline_engine import RedlineEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Legal Document Review AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_analyzers():
    """Initialize and cache the analysis components."""
    try:
        config = get_config()
        parser = DocumentParser()
        legal_analyzer = LegalAnalyzer()
        risk_assessor = RiskAssessor()
        redline_engine = RedlineEngine()
        
        return parser, legal_analyzer, risk_assessor, redline_engine, config
    except Exception as e:
        st.error(f"Failed to initialize analyzers: {e}")
        return None, None, None, None, None

def main():
    """Main Streamlit application."""
    
    # Initialize components
    parser, legal_analyzer, risk_assessor, redline_engine, config = get_analyzers()
    
    if not all([parser, legal_analyzer, risk_assessor, redline_engine, config]):
        st.error("Failed to initialize application components.")
        return
    
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
        
        # API Key Status
        st.markdown("---")
        st.markdown("### API Status")
        if config.llm.anthropic_api_key:
            st.success("✅ Claude API Key configured")
        else:
            st.warning("⚠️ Claude API Key missing")
    
    # Main content based on selected page
    if page == "📄 Document Review":
        document_review_page(parser, legal_analyzer, risk_assessor, redline_engine)
    elif page == "⚡ Quick Risk Assessment":
        quick_risk_page(parser, risk_assessor)
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "⚙️ Settings":
        settings_page(config)


def document_review_page(parser, legal_analyzer, risk_assessor, redline_engine):
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
        
        include_redlining = st.checkbox("Generate Redlined Version", value=True)
        include_precedents = st.checkbox("Research Legal Precedents", value=False)
    
    # Start analysis
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Start Full Analysis", type="primary"):
                run_full_analysis(
                    uploaded_file, document_type, priority_areas, 
                    custom_instructions, include_redlining, include_precedents,
                    parser, legal_analyzer, risk_assessor, redline_engine
                )
        
        with col2:
            if st.button("⚡ Quick Risk Check"):
                run_quick_analysis(uploaded_file, parser, risk_assessor)
        
        # Display file info
        with col3:
            st.info(
                f"**File:** {uploaded_file.name}  \n"
                f"**Size:** {uploaded_file.size / 1024:.1f} KB  \n"
                f"**Type:** {uploaded_file.type}"
            )


def run_full_analysis(uploaded_file, document_type, priority_areas, custom_instructions, 
                     include_redlining, include_precedents, parser, legal_analyzer, 
                     risk_assessor, redline_engine):
    """Run full document analysis with immediate text extraction."""
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        with st.spinner("Extracting text from document..."):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Show immediate text extraction
            status_text.text("📄 Extracting text (including OCR if needed)...")
            progress_bar.progress(10)
            
            # Parse document and show extracted text immediately
            parsed_doc = asyncio.run(parser.parse_file(tmp_file_path))
            
            # Continue with analysis
            status_text.text("🔍 Analyzing legal content...")
            progress_bar.progress(40)
            
            analysis_result = asyncio.run(legal_analyzer.analyze_document(
                parsed_doc, 
                document_type=document_type,
                custom_instructions=custom_instructions
            ))
            
            # Risk assessment
            status_text.text("⚠️ Assessing risks...")
            progress_bar.progress(60)
            
            risk_assessment = asyncio.run(risk_assessor.assess_risks(
                parsed_doc,
                priority_areas=priority_areas
            ))
            
            # Redlining (if requested)
            redlined_content = None
            if include_redlining:
                status_text.text("✏️ Generating redlined version...")
                progress_bar.progress(80)
                
                redlined_content = asyncio.run(redline_engine.generate_redlines(
                    parsed_doc, analysis_result.clause_analysis, risk_assessment
                ))
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            display_analysis_results(analysis_result, risk_assessment, redlined_content)
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.error("Please check your API keys in the .env file and try again.")
        
        # Show debug information
        with st.expander("🔧 Debug Information"):
            st.code(f"Error details: {str(e)}")
            st.info("If this is a text extraction error, the document might be:")
            st.info("• Password protected")
            st.info("• Corrupted")
            st.info("• A scanned image requiring better OCR setup")
            
    finally:
        # Cleanup
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)


def run_quick_analysis(uploaded_file, parser, risk_assessor):
    """Run quick risk assessment."""
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        with st.spinner("Running quick risk assessment..."):
            
            # Parse document
            parsed_doc = asyncio.run(parser.parse_file(tmp_file_path))
            
            # Quick risk assessment
            risk_assessment = asyncio.run(risk_assessor.assess_risks(parsed_doc))
            
            # Display quick results
            display_quick_results(risk_assessment)
            
    except Exception as e:
        st.error(f"Quick analysis failed: {str(e)}")
        st.error("Please check your API keys in the .env file and try again.")
    finally:
        # Cleanup
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)


def display_analysis_results(analysis_result, risk_assessment, redlined_content=None):
    """Display comprehensive analysis results."""
    
    # Create full-width container for results
    st.markdown("---")
    
    # Use custom CSS to make the results section take up more space
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Analysis Results")
    
    # Summary metrics - make them bigger
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_count = len([r for r in risk_assessment.risk_issues if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        st.metric("High/Critical Risks", risk_count)
    
    with col2:
        st.metric("Total Clauses", len(analysis_result.clause_analysis))
    
    with col3:
        avg_enforceability = sum(c.enforceability_score for c in analysis_result.clause_analysis) / len(analysis_result.clause_analysis) if analysis_result.clause_analysis else 0
        st.metric("Avg Enforceability", f"{avg_enforceability:.1f}/100")
    
    with col4:
        overall_score = 100 - (risk_count * 10) + (avg_enforceability / 10)
        st.metric("Overall Score", f"{max(0, min(100, overall_score)):.1f}/100")
    
    # Tabs for different sections
    tabs = st.tabs(["📋 Executive Summary", "⚠️ Risk Assessment", "📜 Clause Analysis", "✏️ Redlined Version"])
    
    with tabs[0]:
        st.subheader("Executive Summary")
        # Use much larger text area for executive summary
        st.text_area(
            "Executive Summary",
            value=analysis_result.executive_summary,
            height=500,  # Increased from 300
            disabled=True,
            label_visibility="collapsed",
            key="executive_summary_text"
        )
        
        if analysis_result.key_findings:
            st.subheader("Key Findings")
            # Create a formatted string for key findings
            key_findings_text = "\n".join([f"• {finding}" for finding in analysis_result.key_findings])
            st.text_area(
                "Key Findings",
                value=key_findings_text,
                height=300,  # Increased from 200
                disabled=True,
                label_visibility="collapsed",
                key="key_findings_text"
            )
    
    with tabs[1]:
        st.subheader("Risk Assessment")
        
        # Risk level distribution
        risk_levels = {}
        for risk in risk_assessment.risk_issues:
            level = risk.risk_level.value
            risk_levels[level] = risk_levels.get(level, 0) + 1
        
        if risk_levels:
            st.bar_chart(risk_levels)
        
        # Individual risks
        for risk in risk_assessment.risk_issues:
            color = {
                "LOW": "green",
                "MEDIUM": "orange", 
                "HIGH": "red",
                "CRITICAL": "darkred"
            }.get(risk.risk_level.value, "gray")
            
            with st.expander(f"🚨 {risk.risk_category} - {risk.risk_level.value}", expanded=risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]):
                st.markdown(f"**Risk:** {risk.description}")
                st.markdown(f"**Impact:** {risk.impact_description}")
                st.markdown(f"**Confidence Score:** {risk.confidence_score:.2f}")
                if risk.mitigation_strategy:
                    st.markdown("**Mitigation Strategy:**")
                    st.write(f"• {risk.mitigation_strategy}")
                if risk.legal_precedents:
                    st.markdown("**Legal Precedents:**")
                    for precedent in risk.legal_precedents:
                        st.write(f"• {precedent}")
    
    with tabs[2]:
        st.subheader("Clause Analysis")
        
        for clause in analysis_result.clause_analysis:
            with st.expander(f"📄 {clause.clause_type}", expanded=False):
                # Show more content with larger text area
                st.text_area(
                    "Clause Content",
                    value=clause.clause_content[:500] + ("..." if len(clause.clause_content) > 500 else ""),
                    height=250,  # Increased from 150
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"clause_content_{clause.clause_id}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Enforceability Score", f"{clause.enforceability_score}/100")
                with col2:
                    st.metric("Business Favorability", f"{clause.business_favorability}/100")
                
                if clause.risk_factors:
                    st.subheader("Risk Factors")
                    risk_factors_text = "\n".join([f"• {issue}" for issue in clause.risk_factors])
                    st.text_area(
                        "Risk Factors",
                        value=risk_factors_text,
                        height=200,  # Increased from 120
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"risk_factors_{clause.clause_id}"
                    )
                    
                if clause.suggested_improvements:
                    st.subheader("Suggested Improvements")
                    improvements_text = "\n".join([f"• {suggestion}" for suggestion in clause.suggested_improvements])
                    st.text_area(
                        "Suggested Improvements",
                        value=improvements_text,
                        height=200,  # Increased from 120
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"improvements_{clause.clause_id}"
                    )
                    
                if clause.legal_reasoning:
                    st.subheader("Legal Reasoning")
                    st.text_area(
                        "Legal Reasoning",
                        value=clause.legal_reasoning,
                        height=250,  # Increased from 150
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"legal_reasoning_{clause.clause_id}"
                    )
    
    with tabs[3]:
        if redlined_content:
            st.subheader("Redlined Version")
            # Use a much larger container for redlined content
            st.markdown(
                f"""
                <div style="height: 800px; overflow-y: auto; border: 1px solid #ddd; padding: 20px; background-color: #fafafa; font-size: 14px; line-height: 1.6;">
                {redlined_content}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Redlining was not requested for this analysis.")


def display_quick_results(risk_assessment):
    """Display quick risk assessment results."""
    
    st.markdown("---")
    st.title("⚡ Quick Risk Assessment")
    
    # Risk summary
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for risk in risk_assessment.risk_issues:
        risk_counts[risk.risk_level.value] += 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🟢 Low", risk_counts["LOW"])
    with col2:
        st.metric("🟡 Medium", risk_counts["MEDIUM"])
    with col3:
        st.metric("🟠 High", risk_counts["HIGH"])
    with col4:
        st.metric("🔴 Critical", risk_counts["CRITICAL"])
    
    # Top risks
    high_risks = [r for r in risk_assessment.risk_issues if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
    
    if high_risks:
        st.subheader("🚨 Priority Risks")
        for risk in high_risks[:5]:  # Show top 5
            st.warning(f"**{risk.risk_category}:** {risk.description}")
    else:
        st.success("✅ No high or critical risks identified!")


def quick_risk_page(parser, risk_assessor):
    """Quick risk assessment page."""
    
    st.title("⚡ Quick Risk Assessment")
    st.markdown("Upload a document for fast risk identification.")
    
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'docx', 'doc', 'txt'],
        key="quick_risk"
    )
    
    if uploaded_file and st.button("🔍 Assess Risks", type="primary"):
        run_quick_analysis(uploaded_file, parser, risk_assessor)


def dashboard_page():
    """Dashboard page."""
    
    st.title("📊 Dashboard")
    st.markdown("Analytics and insights from your document reviews.")
    
    # Placeholder for dashboard content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Reviews")
        st.info("No recent reviews found. Upload a document to get started!")
    
    with col2:
        st.subheader("Risk Trends")
        st.info("Analytics will appear here after processing documents.")


def settings_page(config):
    """Settings page."""
    
    st.title("⚙️ Settings")
    st.markdown("Configure your Legal AI Assistant.")
    
    st.subheader("API Configuration")
    st.info(
        "To use the Legal AI Assistant, you need an API key from Anthropic (Claude). "
        "Edit the `.env` file in the project root to add your key."
    )
    
    st.code("""
# Add this to your .env file:
ANTHROPIC_API_KEY=your_claude_key_here
    """)
    
    st.subheader("Current Configuration")
    st.write(f"**Claude API Key:** {'✅ Configured' if config.llm.anthropic_api_key else '❌ Missing'}")
    
    if st.button("🔄 Reload Configuration"):
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()
