"""
Main Legal Document Review AI Agent.

This is the core orchestration engine that coordinates all components
to perform comprehensive legal document analysis.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.core.models import (
    Document, DocumentType, ReviewRequest, ReviewResult,
    RiskAssessment, ClauseAnalysis, RedlinedDocument,
    ExecutiveSummary, LegalPrecedent
)
from src.core.config import get_config
from src.parsers.document_parser import DocumentParser
from src.analysis.legal_analyzer import LegalAnalyzer
from src.analysis.risk_assessor import RiskAssessor
from src.redlining.redline_engine import RedlineEngine
from src.analysis.precedent_researcher import PrecedentResearcher


class LegalDocumentReviewAgent:
    """
    Main AI agent for legal document review and analysis.
    
    This agent orchestrates the entire review workflow:
    1. Document parsing and structure analysis
    2. Risk assessment and identification
    3. Clause-by-clause legal analysis
    4. Precedent research and case law application
    5. Redlining and change generation
    6. Executive summary creation
    """
    
    def __init__(self):
        """Initialize the legal document review agent."""
        self.config = get_config()
        self.document_parser = DocumentParser()
        self.legal_analyzer = LegalAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.redline_engine = RedlineEngine()
        self.precedent_researcher = PrecedentResearcher()
        
        # Workflow tracking
        self.current_workflow_id: Optional[str] = None
        self.workflow_steps: List[str] = []
        
    async def review_document(
        self, 
        document_path: str, 
        document_type: Optional[DocumentType] = None,
        custom_instructions: Optional[str] = None,
        priority_areas: Optional[List[str]] = None
    ) -> ReviewResult:
        """
        Perform comprehensive legal document review.
        
        Args:
            document_path: Path to the document to review
            document_type: Type of document (if known)
            custom_instructions: Specific instructions for the review
            priority_areas: Areas to focus on during review
            
        Returns:
            Complete review result with all analyses
        """
        start_time = datetime.now()
        workflow_id = str(uuid.uuid4())
        self.current_workflow_id = workflow_id
        self.workflow_steps = []
        
        try:
            # Phase 1: Document Parsing and Analysis
            self._log_workflow_step("Starting document parsing")
            document = await self.document_parser.parse_document(
                document_path, document_type
            )
            
            # Phase 2: Document Classification (if not provided)
            if document.document_type == DocumentType.OTHER:
                self._log_workflow_step("Classifying document type")
                document.document_type = await self.legal_analyzer.classify_document(
                    document.content
                )
            
            # Phase 3: Risk Assessment
            self._log_workflow_step("Performing risk assessment")
            risk_assessment = await self.risk_assessor.assess_risks(
                document, priority_areas or []
            )
            
            # Phase 4: Clause-by-Clause Analysis
            self._log_workflow_step("Analyzing individual clauses")
            clause_analyses = await self.legal_analyzer.analyze_clauses(
                document, risk_assessment
            )
            
            # Phase 5: Precedent Research
            self._log_workflow_step("Researching legal precedents")
            precedents = await self.precedent_researcher.research_precedents(
                document, clause_analyses
            )
            
            # Phase 6: Redlining Generation
            self._log_workflow_step("Generating redlined document")
            redlined_document = await self.redline_engine.generate_redlines(
                document, clause_analyses, risk_assessment
            )
            
            # Phase 7: Executive Summary
            self._log_workflow_step("Creating executive summary")
            executive_summary = await self._create_executive_summary(
                document, risk_assessment, clause_analyses
            )
            
            # Calculate processing time and confidence score
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_confidence_score(
                risk_assessment, clause_analyses
            )
            
            self._log_workflow_step("Review completed successfully")
            
            return ReviewResult(
                document=document,
                risk_assessment=risk_assessment,
                clause_analyses=clause_analyses,
                redlined_document=redlined_document,
                executive_summary=executive_summary,
                legal_precedents=precedents,
                processing_time=processing_time,
                workflow_steps=self.workflow_steps.copy(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self._log_workflow_step(f"Error during review: {str(e)}")
            raise
    
    async def quick_risk_assessment(self, document_path: str) -> RiskAssessment:
        """
        Perform a quick risk assessment without full review.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Risk assessment result
        """
        document = await self.document_parser.parse_document(document_path)
        return await self.risk_assessor.assess_risks(document, [])
    
    async def analyze_specific_clause(
        self, 
        document_path: str, 
        clause_text: str
    ) -> ClauseAnalysis:
        """
        Analyze a specific clause in detail.
        
        Args:
            document_path: Path to the document
            clause_text: Text of the clause to analyze
            
        Returns:
            Detailed clause analysis
        """
        document = await self.document_parser.parse_document(document_path)
        return await self.legal_analyzer.analyze_single_clause(
            clause_text, document
        )
    
    async def compare_documents(
        self, 
        doc1_path: str, 
        doc2_path: str
    ) -> Dict[str, Any]:
        """
        Compare two legal documents and identify differences.
        
        Args:
            doc1_path: Path to first document
            doc2_path: Path to second document
            
        Returns:
            Comparison analysis
        """
        doc1 = await self.document_parser.parse_document(doc1_path)
        doc2 = await self.document_parser.parse_document(doc2_path)
        
        return await self.legal_analyzer.compare_documents(doc1, doc2)
    
    async def _create_executive_summary(
        self,
        document: Document,
        risk_assessment: RiskAssessment,
        clause_analyses: List[ClauseAnalysis]
    ) -> ExecutiveSummary:
        """Create an executive summary for business stakeholders."""
        
        # Extract key risks
        key_risks = [
            issue.title for issue in risk_assessment.critical_issues[:5]
        ]
        
        # Identify critical action items
        critical_actions = []
        for issue in risk_assessment.critical_issues:
            if issue.mitigation_strategy:
                critical_actions.append(issue.mitigation_strategy)
        
        # Determine overall assessment
        if risk_assessment.overall_score >= 80:
            overall_assessment = "HIGH RISK - Immediate attention required"
        elif risk_assessment.overall_score >= 60:
            overall_assessment = "MODERATE RISK - Review recommended"
        elif risk_assessment.overall_score >= 40:
            overall_assessment = "LOW-MODERATE RISK - Minor concerns"
        else:
            overall_assessment = "LOW RISK - Generally acceptable"
        
        # Business impact assessment
        business_impact = await self.legal_analyzer.assess_business_impact(
            document, risk_assessment
        )
        
        # Generate recommendation
        if risk_assessment.overall_score >= 70:
            recommendation = "Recommend legal review before signing"
        elif risk_assessment.overall_score >= 50:
            recommendation = "Consider negotiating key terms"
        else:
            recommendation = "Document appears acceptable with minor modifications"
        
        return ExecutiveSummary(
            document_id=document.id,
            document_type=document.document_type,
            overall_assessment=overall_assessment,
            key_risks=key_risks,
            critical_action_items=critical_actions[:5],
            business_impact=business_impact,
            recommendation=recommendation,
            estimated_review_time="2-4 hours" if risk_assessment.overall_score >= 60 else "1-2 hours"
        )
    
    def _calculate_confidence_score(
        self,
        risk_assessment: RiskAssessment,
        clause_analyses: List[ClauseAnalysis]
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        
        # Base confidence from risk assessment
        base_confidence = 0.8 if len(risk_assessment.risk_issues) > 0 else 0.6
        
        # Adjust based on clause analysis completeness
        if clause_analyses:
            avg_clause_confidence = sum(
                analysis.enforceability_score / 100 
                for analysis in clause_analyses
            ) / len(clause_analyses)
            base_confidence = (base_confidence + avg_clause_confidence) / 2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_confidence))
    
    def _log_workflow_step(self, step: str) -> None:
        """Log a workflow step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        step_with_time = f"[{timestamp}] {step}"
        self.workflow_steps.append(step_with_time)
        print(f"Workflow {self.current_workflow_id}: {step_with_time}")
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a running workflow."""
        if workflow_id == self.current_workflow_id:
            return {
                "workflow_id": workflow_id,
                "status": "running",
                "steps_completed": len(self.workflow_steps),
                "current_step": self.workflow_steps[-1] if self.workflow_steps else "Starting",
                "steps": self.workflow_steps
            }
        else:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "message": "Workflow not found or completed"
            }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id == self.current_workflow_id:
            # Implementation would depend on actual async task management
            self._log_workflow_step("Workflow cancelled by user")
            return True
        return False
