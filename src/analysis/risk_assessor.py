"""
Risk assessment engine for legal documents.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.models import (
    Document, RiskAssessment, RiskIssue, RiskLevel, RiskCategory,
    DocumentClause
)
from src.core.config import get_config
from src.analysis.llm_client import LLMClient


class RiskAssessor:
    """
    AI-powered risk assessment engine for legal documents.
    
    Analyzes documents to identify:
    - Financial risks (payment, damages, liability)
    - Operational risks (performance, delivery, compliance)
    - Legal risks (enforceability, jurisdiction, regulatory)
    - Commercial risks (termination, exclusivity, IP)
    """
    
    def __init__(self):
        """Initialize the risk assessor."""
        self.config = get_config()
        self.llm_client = LLMClient()
        
        # Risk category patterns and weights
        self.risk_patterns = {
            RiskCategory.FINANCIAL: {
                'patterns': [
                    r'\b(unlimited\s+liability|personal\s+guarantee|liquidated\s+damages)\b',
                    r'\b(penalty|fine|forfeit|financial\s+obligation)\b',
                    r'\b(payment\s+default|non[\-\s]payment|overdue)\b',
                    r'\b(indemnif|hold\s+harmless|reimburse)\b'
                ],
                'weight': 1.2,
                'keywords': ['payment', 'cost', 'expense', 'fee', 'penalty', 'damages', 'liability']
            },
            RiskCategory.OPERATIONAL: {
                'patterns': [
                    r'\b(performance\s+standard|service\s+level|delivery\s+schedule)\b',
                    r'\b(compliance\s+requirement|regulatory\s+obligation)\b',
                    r'\b(material\s+breach|substantial\s+default)\b',
                    r'\b(force\s+majeure|act\s+of\s+god|unforeseeable)\b'
                ],
                'weight': 1.0,
                'keywords': ['performance', 'delivery', 'compliance', 'operation', 'breach']
            },
            RiskCategory.LEGAL: {
                'patterns': [
                    r'\b(governing\s+law|jurisdiction|legal\s+proceeding)\b',
                    r'\b(enforceability|validity|legal\s+effect)\b',
                    r'\b(dispute\s+resolution|arbitration|litigation)\b',
                    r'\b(regulatory\s+approval|license\s+requirement)\b'
                ],
                'weight': 1.1,
                'keywords': ['legal', 'court', 'law', 'regulation', 'dispute', 'arbitration']
            },
            RiskCategory.COMMERCIAL: {
                'patterns': [
                    r'\b(termination|expire|end\s+of\s+term)\b',
                    r'\b(exclusivity|non[\-\s]compete|restraint)\b',
                    r'\b(intellectual\s+property|trade\s+secret|confidential)\b',
                    r'\b(assignment|transfer|novation)\b'
                ],
                'weight': 0.9,
                'keywords': ['termination', 'exclusivity', 'assignment', 'intellectual', 'confidential']
            },
            RiskCategory.COMPLIANCE: {
                'patterns': [
                    r'\b(regulatory\s+compliance|legal\s+requirement)\b',
                    r'\b(data\s+protection|privacy\s+law|gdpr)\b',
                    r'\b(anti[\-\s]corruption|money\s+laundering)\b',
                    r'\b(environmental\s+law|safety\s+regulation)\b'
                ],
                'weight': 1.0,
                'keywords': ['compliance', 'regulation', 'privacy', 'safety', 'environmental']
            },
            RiskCategory.INTELLECTUAL_PROPERTY: {
                'patterns': [
                    r'\b(copyright|patent|trademark|trade\s+secret)\b',
                    r'\b(intellectual\s+property|proprietary\s+information)\b',
                    r'\b(license|sublicense|royalty)\b',
                    r'\b(infringement|violation|unauthorized\s+use)\b'
                ],
                'weight': 1.0,
                'keywords': ['copyright', 'patent', 'trademark', 'intellectual', 'proprietary']
            },
            RiskCategory.DATA_PRIVACY: {
                'patterns': [
                    r'\b(personal\s+data|sensitive\s+information|pii)\b',
                    r'\b(data\s+protection|privacy\s+policy|gdpr|ccpa)\b',
                    r'\b(data\s+breach|security\s+incident)\b',
                    r'\b(consent|opt[\-\s]in|opt[\-\s]out)\b'
                ],
                'weight': 1.1,
                'keywords': ['data', 'privacy', 'personal', 'sensitive', 'protection']
            },
            RiskCategory.TERMINATION: {
                'patterns': [
                    r'\b(immediate\s+termination|termination\s+for\s+convenience)\b',
                    r'\b(notice\s+period|cure\s+period|grace\s+period)\b',
                    r'\b(survival\s+clause|post[\-\s]termination)\b',
                    r'\b(wind[\-\s]down|transition\s+period)\b'
                ],
                'weight': 0.8,
                'keywords': ['termination', 'notice', 'cure', 'survival', 'wind-down']
            }
        }
    
    async def assess_risks(
        self, 
        document: Document, 
        priority_areas: List[str] = None
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment of a legal document.
        
        Args:
            document: Document to assess
            priority_areas: Specific risk areas to prioritize
            
        Returns:
            Complete risk assessment
        """
        risk_issues = []
        category_scores = {}
        
        # Pattern-based risk detection
        pattern_risks = await self._detect_pattern_risks(document)
        risk_issues.extend(pattern_risks)
        
        # LLM-based risk analysis
        llm_risks = await self._analyze_risks_with_llm(document, priority_areas)
        risk_issues.extend(llm_risks)
        
        # Clause-specific risk analysis
        clause_risks = await self._analyze_clause_risks(document)
        risk_issues.extend(clause_risks)
        
        # Calculate category scores
        for category in RiskCategory:
            category_risks = [r for r in risk_issues if r.risk_category == category]
            if category_risks:
                # Weight by risk level
                level_weights = {
                    RiskLevel.CRITICAL: 100,
                    RiskLevel.HIGH: 80,
                    RiskLevel.MEDIUM: 60,
                    RiskLevel.LOW: 40,
                    RiskLevel.MINIMAL: 20
                }
                
                total_score = sum(level_weights.get(risk.risk_level, 40) for risk in category_risks)
                avg_score = min(100, total_score // len(category_risks))
                category_scores[category] = avg_score
            else:
                category_scores[category] = 0
        
        # Calculate overall risk score
        overall_score = self._calculate_overall_score(category_scores, priority_areas)
        
        # Identify critical issues
        critical_issues = [
            issue for issue in risk_issues 
            if issue.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ]
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            document, risk_issues, category_scores
        )
        
        return RiskAssessment(
            document_id=document.id,
            overall_score=overall_score,
            risk_issues=risk_issues,
            category_scores=category_scores,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    async def _detect_pattern_risks(self, document: Document) -> List[RiskIssue]:
        """Detect risks using pattern matching."""
        risks = []
        content_lower = document.content.lower()
        
        for category, config in self.risk_patterns.items():
            category_risks = []
            
            # Check for high-risk patterns
            for pattern in config['patterns']:
                import re
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    risk = RiskIssue(
                        id=str(uuid.uuid4()),
                        title=f"{category.value.title()} Risk Identified",
                        description=f"Pattern detected: {match.group()}",
                        risk_level=RiskLevel.MEDIUM,
                        risk_category=category,
                        position=match.start(),
                        impact_description=f"Potential {category.value} exposure",
                        mitigation_strategy=f"Review and negotiate {category.value} terms",
                        confidence_score=0.7
                    )
                    category_risks.append(risk)
            
            # Check for keyword density
            keyword_count = sum(
                content_lower.count(keyword) for keyword in config['keywords']
            )
            
            if keyword_count > 5:  # Threshold for concern
                risk = RiskIssue(
                    id=str(uuid.uuid4()),
                    title=f"High {category.value.title()} Content Detected",
                    description=f"Document contains {keyword_count} {category.value}-related terms",
                    risk_level=RiskLevel.MEDIUM if keyword_count > 10 else RiskLevel.LOW,
                    risk_category=category,
                    impact_description=f"Significant {category.value} implications",
                    mitigation_strategy=f"Detailed review of {category.value} provisions required",
                    confidence_score=0.6
                )
                category_risks.append(risk)
            
            # Limit risks per category
            risks.extend(category_risks[:3])
        
        return risks
    
    async def _analyze_risks_with_llm(
        self, 
        document: Document, 
        priority_areas: Optional[List[str]]
    ) -> List[RiskIssue]:
        """Analyze risks using LLM."""
        risk_prompt = self._build_risk_analysis_prompt(document, priority_areas)
        
        try:
            response = await self.llm_client.generate_response(
                risk_prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            # Parse LLM response into risk issues
            return self._parse_llm_risk_response(response)
            
        except Exception as e:
            print(f"LLM risk analysis failed: {str(e)}")
            return []
    
    async def _analyze_clause_risks(self, document: Document) -> List[RiskIssue]:
        """Analyze risks in specific clauses."""
        risks = []
        
        # Focus on high-importance clauses
        high_risk_clauses = [
            clause for clause in document.clauses
            if clause.importance in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        for clause in high_risk_clauses[:10]:  # Limit to prevent too many API calls
            clause_risks = await self._analyze_single_clause_risk(clause)
            risks.extend(clause_risks)
        
        return risks
    
    async def _analyze_single_clause_risk(self, clause: DocumentClause) -> List[RiskIssue]:
        """Analyze risk in a single clause."""
        clause_prompt = f"""
You are a senior attorney specializing in risk assessment.

Analyze the following contract clause for potential risks:

CLAUSE: {clause.content}
CLAUSE TYPE: {clause.clause_type}

Identify specific risks including:
1. Financial exposure
2. Legal enforceability issues
3. Operational constraints
4. Compliance concerns

For each risk, provide:
- Risk description
- Risk level (Critical/High/Medium/Low)
- Impact description
- Mitigation strategy

Focus on material risks that could significantly impact the client.
"""
        
        try:
            response = await self.llm_client.generate_response(
                clause_prompt,
                max_tokens=800,
                temperature=0.2
            )
            
            return self._parse_clause_risk_response(response, clause)
            
        except Exception as e:
            print(f"Clause risk analysis failed: {str(e)}")
            return []
    
    def _build_risk_analysis_prompt(
        self, 
        document: Document, 
        priority_areas: Optional[List[str]]
    ) -> str:
        """Build prompt for LLM risk analysis."""
        content_preview = (
            document.content[:3000] + "..." 
            if len(document.content) > 3000 
            else document.content
        )
        
        priority_section = ""
        if priority_areas:
            priority_section = f"\nPRIORITY AREAS: Focus especially on {', '.join(priority_areas)}"
        
        return f"""
You are a senior attorney with extensive experience in {document.document_type.value} agreements.

Perform a comprehensive risk assessment of this legal document:

DOCUMENT TYPE: {document.document_type.value}
DOCUMENT CONTENT:
{content_preview}
{priority_section}

Identify the top 5-7 most significant risks across these categories:
1. Financial risks (payment, liability, damages)
2. Operational risks (performance, delivery, compliance)
3. Legal risks (enforceability, jurisdiction)
4. Commercial risks (termination, exclusivity, IP)

For each risk, provide:
- Risk title
- Risk level (Critical/High/Medium/Low)
- Risk category
- Detailed description
- Business impact
- Specific mitigation strategy
- Confidence level (0-100%)

Focus on material risks that could significantly impact the client's business or legal position.
"""
    
    def _parse_llm_risk_response(self, response: str) -> List[RiskIssue]:
        """Parse LLM response into risk issues."""
        risks = []
        
        try:
            # Split response into risk sections
            risk_sections = response.split('\n\n')
            
            for section in risk_sections:
                if len(section.strip()) < 50:
                    continue
                
                # Extract risk information using patterns
                risk_data = self._extract_risk_from_text(section)
                if risk_data:
                    risk = RiskIssue(
                        id=str(uuid.uuid4()),
                        title=risk_data.get('title', 'Identified Risk'),
                        description=risk_data.get('description', section[:200]),
                        risk_level=risk_data.get('level', RiskLevel.MEDIUM),
                        risk_category=risk_data.get('category', RiskCategory.LEGAL),
                        impact_description=risk_data.get('impact', ''),
                        mitigation_strategy=risk_data.get('mitigation', ''),
                        confidence_score=risk_data.get('confidence', 0.7)
                    )
                    risks.append(risk)
        
        except Exception as e:
            print(f"Error parsing LLM risk response: {str(e)}")
        
        return risks[:7]  # Limit to top risks
    
    def _extract_risk_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured risk data from text."""
        import re
        
        # Extract risk level
        level_match = re.search(r'\b(critical|high|medium|low)\b', text, re.IGNORECASE)
        risk_level = RiskLevel.MEDIUM
        if level_match:
            level_str = level_match.group(1).lower()
            if level_str == 'critical':
                risk_level = RiskLevel.CRITICAL
            elif level_str == 'high':
                risk_level = RiskLevel.HIGH
            elif level_str == 'low':
                risk_level = RiskLevel.LOW
        
        # Extract category
        category = RiskCategory.LEGAL  # Default
        for cat in RiskCategory:
            if cat.value in text.lower():
                category = cat
                break
        
        # Extract title (first line or sentence)
        lines = text.strip().split('\n')
        title = lines[0].strip()[:100] if lines else "Risk Identified"
        
        return {
            'title': title,
            'description': text[:500],
            'level': risk_level,
            'category': category,
            'impact': text[:300],
            'mitigation': '',
            'confidence': 0.7
        }
    
    def _parse_clause_risk_response(
        self, 
        response: str, 
        clause: DocumentClause
    ) -> List[RiskIssue]:
        """Parse clause-specific risk response."""
        # Similar to _parse_llm_risk_response but for single clause
        risks = self._parse_llm_risk_response(response)
        
        # Set clause reference for all risks
        for risk in risks:
            risk.clause_id = clause.id
            risk.position = clause.start_position
        
        return risks
    
    def _calculate_overall_score(
        self, 
        category_scores: Dict[RiskCategory, int], 
        priority_areas: Optional[List[str]]
    ) -> int:
        """Calculate overall risk score."""
        if not category_scores:
            return 50  # Default medium risk
        
        # Base score from average
        total_score = sum(category_scores.values())
        avg_score = total_score // len(category_scores)
        
        # Adjust for priority areas
        if priority_areas:
            priority_boost = 0
            for area in priority_areas:
                for category, score in category_scores.items():
                    if area.lower() in category.value.lower():
                        priority_boost += score * 0.2  # 20% boost for priority areas
            
            avg_score = min(100, avg_score + int(priority_boost))
        
        # Apply category weights
        weighted_score = 0
        total_weight = 0
        
        for category, score in category_scores.items():
            weight = self.risk_patterns.get(category, {}).get('weight', 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_score = min(100, int(weighted_score / total_weight))
        
        return max(0, min(100, avg_score))
    
    async def _generate_recommendations(
        self,
        document: Document,
        risk_issues: List[RiskIssue],
        category_scores: Dict[RiskCategory, int]
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        # High-level recommendations based on overall risk
        high_risk_categories = [
            cat for cat, score in category_scores.items() if score >= 70
        ]
        
        if high_risk_categories:
            recommendations.append(
                f"Prioritize negotiation of {', '.join(cat.value for cat in high_risk_categories)} provisions"
            )
        
        # Specific recommendations from critical issues
        critical_issues = [r for r in risk_issues if r.risk_level == RiskLevel.CRITICAL]
        for issue in critical_issues[:3]:
            if issue.mitigation_strategy:
                recommendations.append(issue.mitigation_strategy)
        
        # General recommendations
        if category_scores.get(RiskCategory.FINANCIAL, 0) >= 60:
            recommendations.append("Consider liability caps and indemnification limits")
        
        if category_scores.get(RiskCategory.TERMINATION, 0) >= 60:
            recommendations.append("Negotiate favorable termination notice periods and survival clauses")
        
        return recommendations[:5]  # Limit to top 5 recommendations
