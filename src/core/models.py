"""
Core data models for the Legal Document Review AI Agent.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    """Supported document types for analysis."""
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    MERGER_AGREEMENT = "merger_agreement"
    COMMERCIAL_CONTRACT = "commercial_contract"
    NDA = "nda"
    LICENSE_AGREEMENT = "license_agreement"
    LEASE_AGREEMENT = "lease_agreement"
    SERVICE_AGREEMENT = "service_agreement"
    PARTNERSHIP_AGREEMENT = "partnership_agreement"
    PURCHASE_AGREEMENT = "purchase_agreement"
    LOAN_AGREEMENT = "loan_agreement"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class RiskCategory(str, Enum):
    """Categories of risks in legal documents."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    LEGAL = "legal"
    COMMERCIAL = "commercial"
    COMPLIANCE = "compliance"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PRIVACY = "data_privacy"
    TERMINATION = "termination"


class ChangeType(str, Enum):
    """Types of changes in redlining."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    COMMENT = "comment"
    FORMATTING = "formatting"


class DocumentSection(BaseModel):
    """Represents a section of a legal document."""
    id: str
    title: str
    content: str
    section_number: Optional[str] = None
    subsections: List["DocumentSection"] = Field(default_factory=list)
    start_position: int
    end_position: int
    page_number: Optional[int] = None


class DocumentClause(BaseModel):
    """Represents a specific clause within a document."""
    id: str
    content: str
    clause_type: str
    section_id: str
    importance: RiskLevel
    start_position: int
    end_position: int


class Document(BaseModel):
    """Core document model."""
    id: str
    filename: str
    file_path: str
    document_type: DocumentType
    content: str
    sections: List[DocumentSection] = Field(default_factory=list)
    clauses: List[DocumentClause] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: int
    page_count: Optional[int] = None


class RiskIssue(BaseModel):
    """Represents a specific risk issue identified in the document."""
    id: str
    title: str
    description: str
    risk_level: RiskLevel
    risk_category: RiskCategory
    clause_id: Optional[str] = None
    section_id: Optional[str] = None
    position: Optional[int] = None
    impact_description: str
    mitigation_strategy: str
    legal_precedents: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for a document."""
    document_id: str
    overall_score: int = Field(ge=0, le=100)
    risk_issues: List[RiskIssue] = Field(default_factory=list)
    category_scores: Dict[RiskCategory, int] = Field(default_factory=dict)
    critical_issues: List[RiskIssue] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.now)
    
    def get_issues_by_level(self, level: RiskLevel) -> List[RiskIssue]:
        """Get all issues of a specific risk level."""
        return [issue for issue in self.risk_issues if issue.risk_level == level]


class ClauseAnalysis(BaseModel):
    """Analysis of a specific clause."""
    clause_id: str
    clause_content: str
    content: str = ""  # Alias for clause_content for backward compatibility
    clause_type: str = "Unknown"  # Type/category of this clause
    enforceability_score: int = Field(ge=0, le=100)
    business_favorability: int = Field(ge=0, le=100)
    market_standard_comparison: str
    suggested_improvements: List[str] = Field(default_factory=list)
    legal_reasoning: str
    precedents: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)  # Alias for risk_factors
    
    @model_validator(mode='after')
    def set_aliases(self):
        """Set aliases after validation."""
        if not self.content and self.clause_content:
            self.content = self.clause_content
        if not self.issues and self.risk_factors:
            self.issues = self.risk_factors
        return self


class RedlineChange(BaseModel):
    """Represents a single change in the redlined document."""
    id: str
    change_type: ChangeType
    original_text: str
    suggested_text: str
    position: int
    length: int
    comment: str
    reasoning: str
    risk_level: RiskLevel
    clause_id: Optional[str] = None
    author: str = "AI Legal Assistant"
    timestamp: datetime = Field(default_factory=datetime.now)


class RedlinedDocument(BaseModel):
    """Document with tracked changes and comments."""
    original_document_id: str
    changes: List[RedlineChange] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_changes_by_type(self, change_type: ChangeType) -> List[RedlineChange]:
        """Get all changes of a specific type."""
        return [change for change in self.changes if change.change_type == change_type]


class LegalPrecedent(BaseModel):
    """Represents a legal precedent or case law reference."""
    case_name: str
    citation: str
    jurisdiction: str
    year: int
    relevant_principle: str
    application_to_clause: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class ExecutiveSummary(BaseModel):
    """Business-friendly executive summary."""
    document_id: str
    document_type: DocumentType
    overall_assessment: str
    key_risks: List[str] = Field(default_factory=list)
    critical_action_items: List[str] = Field(default_factory=list)
    business_impact: str
    recommendation: str
    estimated_review_time: str
    created_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Result of legal document analysis."""
    document_id: str
    document_type: Optional[DocumentType] = None
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    clause_analysis: List[ClauseAnalysis] = Field(default_factory=list)
    overall_assessment: str = ""
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)


class ReviewResult(BaseModel):
    """Complete result of document review process."""
    document: Document
    risk_assessment: RiskAssessment
    clause_analyses: List[ClauseAnalysis] = Field(default_factory=list)
    redlined_document: RedlinedDocument
    executive_summary: ExecutiveSummary
    legal_precedents: List[LegalPrecedent] = Field(default_factory=list)
    processing_time: float
    workflow_steps: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)


class ReviewRequest(BaseModel):
    """Request model for document review."""
    document_path: str
    document_type: Optional[DocumentType] = None
    priority_areas: List[RiskCategory] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    custom_instructions: Optional[str] = None
    client_preferences: Dict[str, Any] = Field(default_factory=dict)


# Update forward references
DocumentSection.model_rebuild()
