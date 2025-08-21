"""
Legal document analyzer using LLM for sophisticated legal analysis.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
import json

from src.core.models import (
    Document, DocumentType, ClauseAnalysis, DocumentClause,
    RiskAssessment, LegalPrecedent
)
from src.core.config import get_config
from src.analysis.llm_client import LLMClient


class LegalAnalyzer:
    """
    AI-powered legal document analyzer.
    
    Performs sophisticated legal analysis including:
    - Document classification
    - Clause analysis and improvement suggestions
    - Business impact assessment
    - Legal reasoning and precedent application
    """
    
    def __init__(self):
        """Initialize the legal analyzer."""
        self.config = get_config()
        self.llm_client = LLMClient()
        
        # Document type classification patterns
        self.doc_type_indicators = {
            DocumentType.EMPLOYMENT_AGREEMENT: [
                'employment', 'employee', 'employer', 'position', 'salary',
                'benefits', 'vacation', 'termination of employment'
            ],
            DocumentType.NDA: [
                'non-disclosure', 'confidential information', 'proprietary',
                'trade secrets', 'confidentiality', 'disclosing party'
            ],
            DocumentType.MERGER_AGREEMENT: [
                'merger', 'acquisition', 'acquiring company', 'target company',
                'closing date', 'purchase price', 'due diligence'
            ],
            DocumentType.COMMERCIAL_CONTRACT: [
                'goods', 'services', 'delivery', 'supplier', 'customer',
                'purchase order', 'commercial terms'
            ],
            DocumentType.LICENSE_AGREEMENT: [
                'license', 'licensor', 'licensee', 'intellectual property',
                'royalty', 'grant', 'permitted use'
            ],
            DocumentType.LEASE_AGREEMENT: [
                'lease', 'lessor', 'lessee', 'premises', 'rent', 'tenant',
                'landlord', 'property'
            ],
            DocumentType.SERVICE_AGREEMENT: [
                'services', 'service provider', 'statement of work',
                'deliverables', 'performance', 'service level'
            ]
        }
    
    async def classify_document(self, content: str) -> DocumentType:
        """
        Classify the type of legal document.
        
        Args:
            content: Document content
            
        Returns:
            Classified document type
        """
        # First, try pattern-based classification
        content_lower = content.lower()
        
        type_scores = {}
        for doc_type, indicators in self.doc_type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                type_scores[doc_type] = score
        
        # If clear winner from patterns, return it
        if type_scores:
            max_score = max(type_scores.values())
            if max_score >= 3:  # Threshold for confidence
                return max(type_scores.items(), key=lambda x: x[1])[0]
        
        # Use LLM for classification if patterns are inconclusive
        classification_prompt = self._build_classification_prompt(content)
        
        try:
            response = await self.llm_client.generate_response(
                classification_prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse the response to extract document type
            doc_type = self._parse_classification_response(response)
            return doc_type if doc_type else DocumentType.OTHER
            
        except Exception as e:
            print(f"LLM classification failed: {str(e)}")
            # Fallback to pattern-based classification
            if type_scores:
                return max(type_scores.items(), key=lambda x: x[1])[0]
            return DocumentType.OTHER

    async def analyze_document(
        self,
        document: Document,
        document_type: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> 'AnalysisResult':
        """
        Perform comprehensive legal analysis of a document.
        
        Args:
            document: The parsed document to analyze
            document_type: Optional document type hint
            custom_instructions: Optional custom analysis instructions
            
        Returns:
            Complete analysis result with executive summary and clause analysis
        """
        from src.core.models import AnalysisResult
        
        try:
            # Step 1: Document classification if not provided
            if not document_type:
                document.document_type = await self.classify_document(document.content)
            
            # Step 2: Create a mock risk assessment for clause analysis
            from src.analysis.risk_assessor import RiskAssessor
            risk_assessor = RiskAssessor()
            risk_assessment = await risk_assessor.assess_risks(document)
            
            # Step 3: Analyze clauses
            clause_analyses = await self.analyze_clauses(document, risk_assessment)
            
            # Step 4: Generate executive summary
            executive_summary = await self._generate_executive_summary(
                document, clause_analyses, risk_assessment, custom_instructions
            )
            
            # Step 5: Extract key findings
            key_findings = await self._extract_key_findings(
                document, clause_analyses, risk_assessment
            )
            
            # Create analysis result
            analysis_result = AnalysisResult(
                document_id=document.id,
                document_type=document.document_type,
                executive_summary=executive_summary,
                key_findings=key_findings,
                clause_analysis=clause_analyses,
                overall_assessment=self._generate_overall_assessment(clause_analyses, risk_assessment),
                confidence_score=self._calculate_confidence_score(clause_analyses),
                processing_time=0.0  # Will be set by caller
            )
            
            return analysis_result
            
        except Exception as e:
            print(f"Document analysis failed: {str(e)}")
            # Return minimal analysis result
            return AnalysisResult(
                document_id=document.id,
                document_type=document.document_type or DocumentType.OTHER,
                executive_summary=f"Analysis failed: {str(e)}",
                key_findings=["Document analysis could not be completed"],
                clause_analysis=[],
                overall_assessment="Unable to complete analysis",
                confidence_score=0.0,
                processing_time=0.0
            )
    
    async def analyze_clauses(
        self, 
        document: Document, 
        risk_assessment: RiskAssessment
    ) -> List[ClauseAnalysis]:
        """
        Perform detailed analysis of all clauses in the document.
        
        Args:
            document: Document to analyze
            risk_assessment: Risk assessment for context
            
        Returns:
            List of clause analyses
        """
        analyses = []
        
        # Analyze high-risk clauses first
        high_risk_clauses = [
            clause for clause in document.clauses 
            if clause.importance.value in ['high', 'critical']
        ]
        
        # Then analyze medium and low risk clauses
        other_clauses = [
            clause for clause in document.clauses 
            if clause.importance.value not in ['high', 'critical']
        ]
        
        # Process high-risk clauses with detailed analysis
        for clause in high_risk_clauses:
            analysis = await self.analyze_single_clause(clause.content, document)
            analysis.clause_id = clause.id
            analyses.append(analysis)
        
        # Process other clauses with standard analysis
        for clause in other_clauses[:20]:  # Limit to prevent too many API calls
            analysis = await self.analyze_single_clause(
                clause.content, document, detailed=False
            )
            analysis.clause_id = clause.id
            analyses.append(analysis)
        
        return analyses
    
    async def analyze_single_clause(
        self, 
        clause_content: str, 
        document: Document,
        detailed: bool = True
    ) -> ClauseAnalysis:
        """
        Analyze a single clause in detail.
        
        Args:
            clause_content: Content of the clause
            document: Document context
            detailed: Whether to perform detailed analysis
            
        Returns:
            Clause analysis
        """
        analysis_prompt = self._build_clause_analysis_prompt(
            clause_content, document, detailed
        )
        
        try:
            response = await self.llm_client.generate_response(
                analysis_prompt,
                max_tokens=1000 if detailed else 500,
                temperature=0.2
            )
            
            # Parse the structured response
            analysis_data = self._parse_clause_analysis_response(response)
            
            return ClauseAnalysis(
                clause_id="",  # Will be set by caller
                clause_content=clause_content,
                enforceability_score=analysis_data.get('enforceability_score', 70),
                business_favorability=analysis_data.get('business_favorability', 50),
                market_standard_comparison=analysis_data.get('market_comparison', ''),
                suggested_improvements=analysis_data.get('improvements', []),
                legal_reasoning=analysis_data.get('reasoning', ''),
                precedents=analysis_data.get('precedents', []),
                risk_factors=analysis_data.get('risk_factors', [])
            )
            
        except Exception as e:
            print(f"Clause analysis failed: {str(e)}")
            # Return basic analysis
            return ClauseAnalysis(
                clause_id="",
                clause_content=clause_content,
                enforceability_score=70,
                business_favorability=50,
                market_standard_comparison="Unable to analyze - please review manually",
                legal_reasoning="Analysis unavailable due to technical error"
            )
    
    async def assess_business_impact(
        self, 
        document: Document, 
        risk_assessment: RiskAssessment
    ) -> str:
        """
        Assess the business impact of the document.
        
        Args:
            document: Document to assess
            risk_assessment: Risk assessment for context
            
        Returns:
            Business impact description
        """
        impact_prompt = self._build_business_impact_prompt(document, risk_assessment)
        
        try:
            response = await self.llm_client.generate_response(
                impact_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Business impact assessment failed: {str(e)}")
            return "Business impact assessment unavailable - please review manually"
    
    async def compare_documents(
        self, 
        doc1: Document, 
        doc2: Document
    ) -> Dict[str, Any]:
        """
        Compare two legal documents and identify key differences.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Comparison analysis
        """
        comparison_prompt = self._build_comparison_prompt(doc1, doc2)
        
        try:
            response = await self.llm_client.generate_response(
                comparison_prompt,
                max_tokens=1500,
                temperature=0.2
            )
            
            return self._parse_comparison_response(response)
            
        except Exception as e:
            print(f"Document comparison failed: {str(e)}")
            return {
                "status": "error",
                "message": "Comparison unavailable due to technical error"
            }
    
    def _build_classification_prompt(self, content: str) -> str:
        """Build prompt for document classification."""
        content_preview = content[:2000] + "..." if len(content) > 2000 else content
        
        return f"""
You are a senior attorney specializing in contract law. Analyze the following legal document and classify its type.

Document content:
{content_preview}

Based on the content, classify this document as one of the following types:
- employment_agreement
- merger_agreement
- commercial_contract
- nda (non-disclosure agreement)
- license_agreement
- lease_agreement
- service_agreement
- partnership_agreement
- purchase_agreement
- loan_agreement
- other

Respond with only the document type (no explanation needed).
"""
    
    def _build_clause_analysis_prompt(
        self, 
        clause: str, 
        document: Document, 
        detailed: bool
    ) -> str:
        """Build prompt for clause analysis."""
        analysis_depth = "detailed" if detailed else "standard"
        
        return f"""
You are a senior contract attorney with 15+ years of experience in {document.document_type.value} agreements.

Analyze the following clause for a {document.document_type.value}:

CLAUSE: {clause}

DOCUMENT CONTEXT: {document.document_type.value} agreement

Provide a {analysis_depth} analysis covering:

1. Legal enforceability (score 1-100)
2. Business favorability (score 1-100, where 100 = highly favorable to client)
3. Comparison to market standard practices
4. Specific improvement recommendations
5. Legal reasoning and risk factors
{f'6. Supporting legal precedents' if detailed else ''}

Format your response as a structured analysis with clear sections.
"""
    
    def _build_business_impact_prompt(
        self, 
        document: Document, 
        risk_assessment: RiskAssessment
    ) -> str:
        """Build prompt for business impact assessment."""
        critical_issues = [issue.title for issue in risk_assessment.critical_issues[:3]]
        
        return f"""
You are a business attorney providing executive-level guidance.

Document Type: {document.document_type.value}
Overall Risk Score: {risk_assessment.overall_score}/100

Critical Issues Identified:
{chr(10).join(f"- {issue}" for issue in critical_issues)}

Provide a business impact assessment in 2-3 sentences covering:
1. Primary business risks and opportunities
2. Operational implications
3. Financial exposure or benefits
4. Strategic considerations

Write in clear, business-friendly language for executive stakeholders.
"""
    
    def _build_comparison_prompt(self, doc1: Document, doc2: Document) -> str:
        """Build prompt for document comparison."""
        content1_preview = doc1.content[:1000] + "..." if len(doc1.content) > 1000 else doc1.content
        content2_preview = doc2.content[:1000] + "..." if len(doc2.content) > 1000 else doc2.content
        
        return f"""
You are a senior attorney comparing two legal documents.

Document 1 ({doc1.filename}):
{content1_preview}

Document 2 ({doc2.filename}):
{content2_preview}

Identify and analyze:
1. Key differences in terms and conditions
2. Risk profile changes
3. Business impact of differences
4. Recommendations for negotiation

Provide a structured comparison focusing on material differences.
"""
    
    def _parse_classification_response(self, response: str) -> Optional[DocumentType]:
        """Parse the document classification response."""
        response_lower = response.lower().strip()
        
        # Map response to DocumentType enum
        type_mapping = {
            'employment_agreement': DocumentType.EMPLOYMENT_AGREEMENT,
            'merger_agreement': DocumentType.MERGER_AGREEMENT,
            'commercial_contract': DocumentType.COMMERCIAL_CONTRACT,
            'nda': DocumentType.NDA,
            'license_agreement': DocumentType.LICENSE_AGREEMENT,
            'lease_agreement': DocumentType.LEASE_AGREEMENT,
            'service_agreement': DocumentType.SERVICE_AGREEMENT,
            'partnership_agreement': DocumentType.PARTNERSHIP_AGREEMENT,
            'purchase_agreement': DocumentType.PURCHASE_AGREEMENT,
            'loan_agreement': DocumentType.LOAN_AGREEMENT,
            'other': DocumentType.OTHER
        }
        
        return type_mapping.get(response_lower)

    async def _generate_executive_summary(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Generate an executive summary of the document analysis."""
        try:
            high_risk_clauses = [c for c in clause_analyses if c.issues]
            total_clauses = len(clause_analyses)
            avg_enforceability = sum(c.enforceability_score for c in clause_analyses) / total_clauses if total_clauses > 0 else 0
            
            prompt = f"""
Provide an executive summary for this legal document analysis:

Document Type: {document.document_type.value if document.document_type else 'Unknown'}
Total Clauses Analyzed: {total_clauses}
Average Enforceability Score: {avg_enforceability:.1f}/10
High-Risk Clauses: {len(high_risk_clauses)}
Total Risk Items: {len(risk_assessment.risk_issues) if risk_assessment else 0}

Key Issues Found:
{chr(10).join([f"- {issue}" for clause in clause_analyses for issue in clause.risk_factors[:2]])}

Custom Instructions: {custom_instructions or "None"}

Provide a concise executive summary (2-3 paragraphs) suitable for business stakeholders, focusing on:
1. Overall document quality and enforceability
2. Key risks and business implications
3. Recommended next steps
"""
            
            response = await self.llm_client.generate_response(prompt)
            return response.strip()
            
        except Exception as e:
            return f"Executive summary generation failed: {str(e)}"

    async def _extract_key_findings(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        try:
            # Critical issues from clauses
            critical_issues = [issue for clause in clause_analyses for issue in clause.risk_factors if 'critical' in issue.lower()]
            findings.extend(critical_issues[:3])  # Top 3 critical issues
            
            # High-risk items
            if risk_assessment:
                high_risks = [risk.description for risk in risk_assessment.risk_issues if risk.risk_level.value in ['HIGH', 'CRITICAL']]
                findings.extend(high_risks[:3])  # Top 3 high risks
            
            # Enforceability concerns
            low_enforceability = [f"Low enforceability in {clause.clause_type}" 
                                for clause in clause_analyses if clause.enforceability_score < 5]
            findings.extend(low_enforceability[:2])  # Top 2 enforceability concerns
            
            return findings[:8]  # Return top 8 findings
            
        except Exception as e:
            return [f"Key findings extraction failed: {str(e)}"]

    def _generate_overall_assessment(
        self,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment
    ) -> str:
        """Generate an overall assessment of the document."""
        try:
            if not clause_analyses:
                return "No clauses were analyzed"
            
            avg_enforceability = sum(c.enforceability_score for c in clause_analyses) / len(clause_analyses)
            total_issues = sum(len(c.issues) for c in clause_analyses)
            high_risk_count = len([r for r in risk_assessment.risk_issues if r.risk_level.value in ['HIGH', 'CRITICAL']]) if risk_assessment else 0
            
            if avg_enforceability >= 8 and total_issues <= 2 and high_risk_count == 0:
                return "Excellent - Well-drafted document with minimal issues"
            elif avg_enforceability >= 6 and total_issues <= 5 and high_risk_count <= 2:
                return "Good - Solid document with minor issues that should be addressed"
            elif avg_enforceability >= 4 and total_issues <= 10 and high_risk_count <= 5:
                return "Fair - Document needs significant review and improvements"
            else:
                return "Poor - Document has major issues requiring substantial revision"
                
        except Exception as e:
            return f"Assessment generation failed: {str(e)}"

    def _calculate_confidence_score(self, clause_analyses: List[ClauseAnalysis]) -> float:
        """Calculate confidence score for the analysis."""
        try:
            if not clause_analyses:
                return 0.0
            
            # Base confidence on number of clauses analyzed and completeness
            base_score = min(len(clause_analyses) / 10.0, 1.0)  # More clauses = higher confidence
            
            # Adjust based on enforceability scores (more consistent scores = higher confidence)
            scores = [c.enforceability_score for c in clause_analyses]
            if scores:
                score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
                consistency_factor = max(0, 1 - score_variance / 25)  # Normalize variance
                return min(base_score * consistency_factor, 1.0)
            
            return base_score
            
        except Exception as e:
            return 0.5  # Default moderate confidence
    
    def _parse_clause_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the clause analysis response."""
        analysis_data = {
            'enforceability_score': 70,
            'business_favorability': 50,
            'market_comparison': '',
            'improvements': [],
            'reasoning': '',
            'precedents': [],
            'risk_factors': []
        }
        
        try:
            # Extract scores using regex
            enforceability_match = re.search(r'enforceability.*?(\d+)', response, re.IGNORECASE)
            if enforceability_match:
                analysis_data['enforceability_score'] = int(enforceability_match.group(1))
            
            favorability_match = re.search(r'favorability.*?(\d+)', response, re.IGNORECASE)
            if favorability_match:
                analysis_data['business_favorability'] = int(favorability_match.group(1))
            
            # Extract text sections
            analysis_data['reasoning'] = response
            analysis_data['market_comparison'] = response
            
            # Extract recommendations (lines starting with "recommend", "suggest", etc.)
            recommendations = re.findall(
                r'(?:recommend|suggest|improve|consider)[^.]*\.', 
                response, 
                re.IGNORECASE
            )
            analysis_data['improvements'] = recommendations[:5]  # Limit to 5
            
        except Exception as e:
            print(f"Error parsing clause analysis: {str(e)}")
        
        return analysis_data
    
    def _parse_comparison_response(self, response: str) -> Dict[str, Any]:
        """Parse the document comparison response."""
        return {
            "summary": response,
            "key_differences": [],
            "risk_changes": [],
            "recommendations": []
            # TODO: Implement structured parsing of comparison results
        }
