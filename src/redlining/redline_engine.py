"""
Redlining engine for generating tracked changes and comments.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from src.core.models import (
    Document, RedlinedDocument, RedlineChange, ChangeType, RiskLevel,
    ClauseAnalysis, RiskAssessment, RiskIssue
)
from src.core.config import get_config
from src.analysis.llm_client import LLMClient


class RedlineEngine:
    """
    AI-powered redlining engine for legal documents.
    
    Generates tracked changes and detailed comments based on:
    - Risk assessment findings
    - Clause analysis recommendations
    - Legal best practices
    - Market standard terms
    """
    
    def __init__(self):
        """Initialize the redlining engine."""
        self.config = get_config()
        self.llm_client = LLMClient()
        
        # Standard improvement patterns
        self.improvement_patterns = {
            'liability_caps': {
                'pattern': r'\b(unlimited\s+liability|unlimited\s+damages)\b',
                'suggestion': 'Consider adding liability caps to limit exposure',
                'risk_level': RiskLevel.HIGH
            },
            'force_majeure': {
                'pattern': r'\b(force\s+majeure)\b',
                'suggestion': 'Expand force majeure clause to include pandemics and cyber events',
                'risk_level': RiskLevel.MEDIUM
            },
            'termination_notice': {
                'pattern': r'\b(immediate\s+termination|terminate\s+immediately)\b',
                'suggestion': 'Add reasonable notice period for termination',
                'risk_level': RiskLevel.MEDIUM
            },
            'governing_law': {
                'pattern': r'\b(governing\s+law)\b',
                'suggestion': 'Ensure governing law jurisdiction is favorable',
                'risk_level': RiskLevel.MEDIUM
            },
            'indemnification': {
                'pattern': r'\b(indemnify|hold\s+harmless)\b',
                'suggestion': 'Consider mutual indemnification or carve-outs',
                'risk_level': RiskLevel.HIGH
            }
        }
    
    async def generate_redlines(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment
    ) -> RedlinedDocument:
        """
        Generate redlined document with tracked changes and comments.
        
        Args:
            document: Original document
            clause_analyses: Detailed clause analyses
            risk_assessment: Risk assessment results
            
        Returns:
            Redlined document with changes and comments
        """
        changes = []
        
        # Generate changes based on risk issues
        risk_changes = await self._generate_risk_based_changes(
            document, risk_assessment
        )
        changes.extend(risk_changes)
        
        # Generate changes based on clause analysis
        clause_changes = await self._generate_clause_based_changes(
            document, clause_analyses
        )
        changes.extend(clause_changes)
        
        # Generate pattern-based improvements
        pattern_changes = await self._generate_pattern_based_changes(document)
        changes.extend(pattern_changes)
        
        # Generate LLM-based improvements
        llm_changes = await self._generate_llm_based_changes(
            document, clause_analyses, risk_assessment
        )
        changes.extend(llm_changes)
        
        # Sort changes by position
        changes.sort(key=lambda x: x.position)
        
        # Generate summary comments
        comments = await self._generate_summary_comments(
            document, risk_assessment, len(changes)
        )
        
        return RedlinedDocument(
            original_document_id=document.id,
            changes=changes,
            comments=comments
        )
    
    async def _generate_risk_based_changes(
        self,
        document: Document,
        risk_assessment: RiskAssessment
    ) -> List[RedlineChange]:
        """Generate changes based on identified risks."""
        changes = []
        
        for risk in risk_assessment.critical_issues[:5]:  # Focus on top critical issues
            if risk.position and risk.mitigation_strategy:
                # Find the problematic text around the risk position
                start_pos = max(0, risk.position - 100)
                end_pos = min(len(document.content), risk.position + 100)
                context = document.content[start_pos:end_pos]
                
                # Generate improvement suggestion
                change = await self._create_improvement_change(
                    context, risk, start_pos
                )
                if change:
                    changes.append(change)
        
        return changes
    
    async def _generate_clause_based_changes(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis]
    ) -> List[RedlineChange]:
        """Generate changes based on clause analysis recommendations."""
        changes = []
        
        for analysis in clause_analyses:
            if analysis.suggested_improvements:
                # Find the clause in the document
                clause = self._find_clause_by_id(document, analysis.clause_id)
                if clause:
                    for improvement in analysis.suggested_improvements[:2]:  # Limit per clause
                        change = RedlineChange(
                            id=str(uuid.uuid4()),
                            change_type=ChangeType.COMMENT,
                            original_text=clause.content[:100] + "...",
                            suggested_text="",
                            position=clause.start_position,
                            length=len(clause.content),
                            comment=improvement,
                            reasoning=analysis.legal_reasoning[:200],
                            risk_level=self._assess_change_risk_level(analysis),
                            clause_id=clause.id
                        )
                        changes.append(change)
        
        return changes
    
    async def _generate_pattern_based_changes(
        self, 
        document: Document
    ) -> List[RedlineChange]:
        """Generate changes based on common improvement patterns."""
        changes = []
        content = document.content
        
        for pattern_name, pattern_config in self.improvement_patterns.items():
            matches = re.finditer(
                pattern_config['pattern'], 
                content, 
                re.IGNORECASE
            )
            
            for match in matches:
                change = RedlineChange(
                    id=str(uuid.uuid4()),
                    change_type=ChangeType.COMMENT,
                    original_text=match.group(),
                    suggested_text="",
                    position=match.start(),
                    length=len(match.group()),
                    comment=pattern_config['suggestion'],
                    reasoning=f"Standard improvement for {pattern_name.replace('_', ' ')}",
                    risk_level=pattern_config['risk_level']
                )
                changes.append(change)
        
        return changes
    
    async def _generate_llm_based_changes(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment
    ) -> List[RedlineChange]:
        """Generate sophisticated changes using LLM analysis."""
        redline_prompt = self._build_redline_prompt(
            document, clause_analyses, risk_assessment
        )
        
        try:
            response = await self.llm_client.generate_response(
                redline_prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            return self._parse_llm_redline_response(response, document)
            
        except Exception as e:
            print(f"LLM redlining failed: {str(e)}")
            return []
    
    def _build_redline_prompt(
        self,
        document: Document,
        clause_analyses: List[ClauseAnalysis],
        risk_assessment: RiskAssessment
    ) -> str:
        """Build prompt for LLM-based redlining."""
        
        # Extract key issues
        critical_issues = [
            issue.title for issue in risk_assessment.critical_issues[:3]
        ]
        
        # Extract low-scoring clauses
        problematic_clauses = [
            analysis for analysis in clause_analyses
            if analysis.enforceability_score < 70 or analysis.business_favorability < 60
        ]
        
        content_preview = (
            document.content[:2000] + "..." 
            if len(document.content) > 2000 
            else document.content
        )
        
        return f"""
You are a senior contract attorney reviewing a {document.document_type.value} for redlining.

DOCUMENT CONTENT:
{content_preview}

CRITICAL ISSUES IDENTIFIED:
{chr(10).join(f"- {issue}" for issue in critical_issues)}

PROBLEMATIC CLAUSES:
{chr(10).join(f"- {clause.clause_content[:100]}..." for clause in problematic_clauses[:3])}

Generate specific redline suggestions including:

1. ADDITIONS: New language to add for protection
2. DELETIONS: Problematic language to remove
3. MODIFICATIONS: Existing language to revise
4. COMMENTS: Explanatory notes for negotiation

For each suggestion, provide:
- Specific text location (quote the exact text)
- Type of change (add/delete/modify/comment)
- Suggested revision
- Business reasoning
- Risk level (Critical/High/Medium/Low)

Focus on the most impactful changes that will materially improve the client's position.
Provide 5-7 specific, actionable redline suggestions.
"""
    
    def _parse_llm_redline_response(
        self, 
        response: str, 
        document: Document
    ) -> List[RedlineChange]:
        """Parse LLM response into redline changes."""
        changes = []
        
        try:
            # Split response into suggestion blocks
            suggestion_blocks = re.split(r'\n\s*\d+\.', response)
            
            for block in suggestion_blocks:
                if len(block.strip()) < 50:
                    continue
                
                change_data = self._extract_change_from_block(block, document)
                if change_data:
                    change = RedlineChange(
                        id=str(uuid.uuid4()),
                        change_type=change_data['type'],
                        original_text=change_data['original_text'],
                        suggested_text=change_data['suggested_text'],
                        position=change_data['position'],
                        length=change_data['length'],
                        comment=change_data['comment'],
                        reasoning=change_data['reasoning'],
                        risk_level=change_data['risk_level']
                    )
                    changes.append(change)
        
        except Exception as e:
            print(f"Error parsing LLM redline response: {str(e)}")
        
        return changes[:7]  # Limit to top suggestions
    
    def _extract_change_from_block(
        self, 
        block: str, 
        document: Document
    ) -> Optional[Dict[str, Any]]:
        """Extract change information from text block."""
        
        # Determine change type
        change_type = ChangeType.COMMENT  # Default
        if 'add' in block.lower() or 'insert' in block.lower():
            change_type = ChangeType.ADDITION
        elif 'delete' in block.lower() or 'remove' in block.lower():
            change_type = ChangeType.DELETION
        elif 'modify' in block.lower() or 'revise' in block.lower():
            change_type = ChangeType.MODIFICATION
        
        # Extract quoted text (original text)
        quote_pattern = r'"([^"]+)"'
        quote_matches = re.findall(quote_pattern, block)
        original_text = quote_matches[0] if quote_matches else ""
        
        # Find position in document
        position = 0
        length = 0
        if original_text:
            try:
                position = document.content.find(original_text)
                length = len(original_text)
            except:
                pass
        
        # Extract suggested text
        suggested_text = ""
        if change_type in [ChangeType.ADDITION, ChangeType.MODIFICATION]:
            # Look for suggested replacement text
            suggest_patterns = [
                r'suggest[^:]*:\s*"([^"]+)"',
                r'replace.*with[^:]*:\s*"([^"]+)"',
                r'revise.*to[^:]*:\s*"([^"]+)"'
            ]
            
            for pattern in suggest_patterns:
                match = re.search(pattern, block, re.IGNORECASE)
                if match:
                    suggested_text = match.group(1)
                    break
        
        # Determine risk level
        risk_level = RiskLevel.MEDIUM  # Default
        if 'critical' in block.lower():
            risk_level = RiskLevel.CRITICAL
        elif 'high' in block.lower():
            risk_level = RiskLevel.HIGH
        elif 'low' in block.lower():
            risk_level = RiskLevel.LOW
        
        return {
            'type': change_type,
            'original_text': original_text or block[:100],
            'suggested_text': suggested_text,
            'position': position,
            'length': length,
            'comment': block[:300],
            'reasoning': block[:200],
            'risk_level': risk_level
        }
    
    async def _create_improvement_change(
        self,
        context: str,
        risk: RiskIssue,
        position: int
    ) -> Optional[RedlineChange]:
        """Create an improvement change for a specific risk."""
        
        # Generate specific improvement using LLM
        improvement_prompt = f"""
You are a contract attorney. Provide a specific redline suggestion for this risk:

CONTEXT: {context}
RISK: {risk.description}
MITIGATION: {risk.mitigation_strategy}

Provide:
1. Exact text to modify (quote the problematic language)
2. Suggested replacement text
3. Brief explanation of the improvement

Be specific and actionable.
"""
        
        try:
            response = await self.llm_client.generate_response(
                improvement_prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            # Parse response for specific change
            quote_match = re.search(r'"([^"]+)"', response)
            original_text = quote_match.group(1) if quote_match else context[:50]
            
            return RedlineChange(
                id=str(uuid.uuid4()),
                change_type=ChangeType.MODIFICATION,
                original_text=original_text,
                suggested_text="[See comment for suggested revision]",
                position=position,
                length=len(original_text),
                comment=response,
                reasoning=risk.mitigation_strategy,
                risk_level=risk.risk_level
            )
            
        except Exception as e:
            print(f"Failed to create improvement change: {str(e)}")
            return None
    
    def _find_clause_by_id(self, document: Document, clause_id: str):
        """Find a clause by its ID."""
        for clause in document.clauses:
            if clause.id == clause_id:
                return clause
        return None
    
    def _assess_change_risk_level(self, analysis: ClauseAnalysis) -> RiskLevel:
        """Assess the risk level of a clause change."""
        if analysis.enforceability_score < 50:
            return RiskLevel.HIGH
        elif analysis.business_favorability < 40:
            return RiskLevel.HIGH
        elif analysis.enforceability_score < 70:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _generate_summary_comments(
        self,
        document: Document,
        risk_assessment: RiskAssessment,
        change_count: int
    ) -> List[str]:
        """Generate summary comments for the redlined document."""
        comments = []
        
        # Overall assessment comment
        if risk_assessment.overall_score >= 70:
            comments.append(
                f"HIGH RISK DOCUMENT: Overall risk score {risk_assessment.overall_score}/100. "
                f"Immediate attention required before signing."
            )
        elif risk_assessment.overall_score >= 50:
            comments.append(
                f"MODERATE RISK: Overall risk score {risk_assessment.overall_score}/100. "
                f"Review and negotiate key terms recommended."
            )
        else:
            comments.append(
                f"LOW-MODERATE RISK: Overall risk score {risk_assessment.overall_score}/100. "
                f"Document generally acceptable with minor modifications."
            )
        
        # Critical issues comment
        if risk_assessment.critical_issues:
            critical_count = len(risk_assessment.critical_issues)
            comments.append(
                f"CRITICAL ISSUES: {critical_count} critical issues identified requiring "
                f"immediate attention: {', '.join(issue.title for issue in risk_assessment.critical_issues[:3])}"
            )
        
        # Changes summary
        if change_count > 0:
            comments.append(
                f"REDLINE SUMMARY: {change_count} changes and comments generated. "
                f"Review all tracked changes and comments before proceeding."
            )
        
        # Recommendations
        if risk_assessment.recommendations:
            comments.append(
                f"KEY RECOMMENDATIONS: {'; '.join(risk_assessment.recommendations[:3])}"
            )
        
        return comments
