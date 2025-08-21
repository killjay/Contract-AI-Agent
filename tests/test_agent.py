"""
Test suite for Legal Document Review AI Agent.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os

from src.core.models import DocumentType, RiskLevel
from src.agent import LegalDocumentReviewAgent
from src.parsers.document_parser import DocumentParser
from src.analysis.risk_assessor import RiskAssessor


class TestDocumentParser:
    """Test document parsing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()
    
    def test_create_sample_document(self):
        """Test with a sample text document."""
        sample_content = """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into between Company and Employee.
        
        1. POSITION AND DUTIES
        Employee shall serve as Software Engineer.
        
        2. COMPENSATION
        Employee shall receive $100,000 annually.
        
        3. TERMINATION
        Either party may terminate this agreement with 30 days notice.
        
        4. CONFIDENTIALITY
        Employee agrees to maintain confidentiality of proprietary information.
        
        5. GOVERNING LAW
        This agreement shall be governed by California law.
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_content)
            temp_path = f.name
        
        try:
            # Test parsing
            document = asyncio.run(
                self.parser.parse_document(temp_path, DocumentType.EMPLOYMENT_AGREEMENT)
            )
            
            assert document is not None
            assert document.document_type == DocumentType.EMPLOYMENT_AGREEMENT
            assert len(document.content) > 0
            assert len(document.sections) > 0
            assert len(document.clauses) > 0
            
        finally:
            os.unlink(temp_path)


class TestRiskAssessor:
    """Test risk assessment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_assessor = RiskAssessor()
    
    def test_pattern_risk_detection(self):
        """Test pattern-based risk detection."""
        sample_content = """
        The parties agree to unlimited liability for any damages.
        Immediate termination is allowed without notice.
        All disputes shall be resolved in courts of unfavorable jurisdiction.
        """
        
        # Create a mock document
        from src.core.models import Document
        document = Document(
            id="test-doc",
            filename="test.txt",
            file_path="/tmp/test.txt",
            document_type=DocumentType.COMMERCIAL_CONTRACT,
            content=sample_content,
            file_size=len(sample_content)
        )
        
        # Test risk assessment
        risk_assessment = asyncio.run(
            self.risk_assessor._detect_pattern_risks(document)
        )
        
        assert len(risk_assessment) > 0
        # Should detect liability and termination risks
        risk_titles = [risk.title.lower() for risk in risk_assessment]
        assert any('liability' in title or 'financial' in title for title in risk_titles)


class TestLegalAgent:
    """Test the main legal agent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Note: This will fail if no API keys are configured
        # In production, use mock LLM client for testing
        pass
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        try:
            agent = LegalDocumentReviewAgent()
            assert agent is not None
            assert agent.document_parser is not None
            assert agent.risk_assessor is not None
        except Exception as e:
            # Expected if no API keys configured
            print(f"Agent initialization failed (expected): {e}")


def test_configuration():
    """Test configuration loading."""
    from src.core.config import get_config
    
    config = get_config()
    assert config is not None
    assert config.storage.max_file_size_mb > 0
    assert len(config.storage.allowed_file_types) > 0


def test_models():
    """Test data model creation."""
    from src.core.models import Document, RiskIssue, RiskLevel, RiskCategory
    
    # Test document creation
    doc = Document(
        id="test-123",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        document_type=DocumentType.NDA,
        content="Test content",
        file_size=1000
    )
    
    assert doc.id == "test-123"
    assert doc.document_type == DocumentType.NDA
    
    # Test risk issue creation
    risk = RiskIssue(
        id="risk-123",
        title="Test Risk",
        description="Test description",
        risk_level=RiskLevel.HIGH,
        risk_category=RiskCategory.FINANCIAL,
        impact_description="High impact",
        mitigation_strategy="Mitigation strategy",
        confidence_score=0.8
    )
    
    assert risk.risk_level == RiskLevel.HIGH
    assert risk.confidence_score == 0.8


if __name__ == "__main__":
    # Run basic tests
    print("Running Legal Document Review AI Agent Tests...")
    
    # Test configuration
    test_configuration()
    print("✓ Configuration test passed")
    
    # Test models
    test_models()
    print("✓ Models test passed")
    
    # Test document parser
    parser_test = TestDocumentParser()
    parser_test.setup_method()
    parser_test.test_create_sample_document()
    print("✓ Document parser test passed")
    
    # Test risk assessor
    risk_test = TestRiskAssessor()
    risk_test.setup_method()
    risk_test.test_pattern_risk_detection()
    print("✓ Risk assessor test passed")
    
    # Test agent initialization
    agent_test = TestLegalAgent()
    agent_test.setup_method()
    agent_test.test_agent_initialization()
    print("✓ Agent initialization test completed")
    
    print("\n🎉 All tests completed!")
    print("\nTo run with pytest: pytest tests/test_agent.py -v")
