# Legal Document Review AI Agent - Usage Guide

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add your API keys
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 2. Run the Application

#### Option A: Web Interface (Recommended)
```bash
streamlit run src/ui/app.py
```
Or use the batch file: `start_ui.bat`

#### Option B: API Server
```bash
python -m src.api.main
```
Or use the batch file: `start_api.bat`

#### Option C: Python SDK
```python
from src.agent import LegalDocumentReviewAgent

agent = LegalDocumentReviewAgent()
result = await agent.review_document("path/to/contract.pdf")
```

## 📋 Features

### 🔍 Document Analysis
- **Multi-format Support**: PDF, Word (.docx, .doc), Text files
- **Document Classification**: Automatic detection of contract types
- **Structure Analysis**: Sections, clauses, and legal provisions
- **Content Extraction**: Preserves formatting and metadata

### ⚠️ Risk Assessment
- **Comprehensive Scoring**: 0-100 risk scale across categories
- **Risk Categories**: Financial, Operational, Legal, Commercial, Compliance, IP, Data Privacy, Termination
- **Pattern Detection**: Identifies problematic clauses using legal patterns
- **AI Analysis**: Advanced risk evaluation using LLM reasoning

### 📝 Clause Analysis
- **Enforceability Scoring**: Legal validity assessment (0-100)
- **Business Favorability**: Terms favorable to client (0-100)
- **Market Comparison**: Comparison to standard market practices
- **Improvement Suggestions**: Specific recommendations for better terms

### ✏️ Redlining Engine
- **Tracked Changes**: Professional redlining with change tracking
- **Detailed Comments**: Explanations for each suggested change
- **Risk-Based Prioritization**: Critical changes highlighted first
- **Legal Reasoning**: Citations and legal basis for changes

### 📚 Legal Research
- **Precedent Search**: Relevant case law and legal precedents
- **Multi-Database**: Google Scholar, free legal databases
- **Contextual Relevance**: AI-powered relevance scoring
- **Application Guidance**: How precedents apply to specific clauses

### 📊 Executive Reporting
- **Business-Friendly Summaries**: Non-legal language for executives
- **Risk Highlights**: Key risks and critical action items
- **Processing Statistics**: Confidence scores and analysis metrics
- **Actionable Recommendations**: Specific next steps

## 🎯 Use Cases

### 1. Contract Review
```python
# Comprehensive contract analysis
result = await agent.review_document(
    document_path="employment_agreement.pdf",
    document_type=DocumentType.EMPLOYMENT_AGREEMENT,
    priority_areas=["financial", "termination", "intellectual_property"],
    custom_instructions="Focus on employee protection and fair terms"
)

print(f"Risk Score: {result.risk_assessment.overall_score}/100")
print(f"Critical Issues: {len(result.risk_assessment.critical_issues)}")
```

### 2. Quick Risk Check
```python
# Fast risk assessment
risk_assessment = await agent.quick_risk_assessment("contract.pdf")
print(f"Overall Risk: {risk_assessment.overall_score}/100")
```

### 3. Clause Analysis
```python
# Analyze specific clause
analysis = await agent.analyze_specific_clause(
    document_path="contract.pdf",
    clause_text="Either party may terminate this agreement immediately..."
)
print(f"Enforceability: {analysis.enforceability_score}%")
```

### 4. Document Comparison
```python
# Compare two versions
comparison = await agent.compare_documents("v1.pdf", "v2.pdf")
print(comparison["summary"])
```

## 🔧 Configuration

### Environment Variables
```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_LLM=claude-3-sonnet

# Application Settings
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,docx,doc,txt

# Risk Assessment Thresholds
CRITICAL_RISK_THRESHOLD=80
HIGH_RISK_THRESHOLD=60
MEDIUM_RISK_THRESHOLD=40
```

### API Configuration
```python
# Custom configuration
from src.core.config import update_config

update_config(
    llm_temperature=0.1,
    max_concurrent_reviews=5,
    document_chunk_size=1000
)
```

## 📖 API Reference

### REST API Endpoints

#### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <document_file>
```

#### Start Analysis
```http
POST /analyze
Content-Type: application/json

{
  "file_id": "uuid",
  "document_type": "employment_agreement",
  "priority_areas": ["financial", "termination"],
  "custom_instructions": "Focus on specific areas..."
}
```

#### Check Status
```http
GET /status/{workflow_id}
```

#### Get Results
```http
GET /result/{workflow_id}
```

#### Quick Risk Assessment
```http
POST /quick-risk-assessment
Content-Type: application/json

{
  "file_id": "uuid"
}
```

### Python SDK

#### Initialize Agent
```python
from src.agent import LegalDocumentReviewAgent

agent = LegalDocumentReviewAgent()
```

#### Review Document
```python
result = await agent.review_document(
    document_path="contract.pdf",
    document_type=DocumentType.COMMERCIAL_CONTRACT,
    priority_areas=["liability", "termination"],
    custom_instructions="Client-specific requirements"
)
```

#### Access Results
```python
# Risk assessment
print(f"Overall Risk: {result.risk_assessment.overall_score}")
for issue in result.risk_assessment.critical_issues:
    print(f"Critical: {issue.title}")

# Clause analyses
for analysis in result.clause_analyses:
    print(f"Clause: {analysis.enforceability_score}% enforceable")

# Redlined changes
for change in result.redlined_document.changes:
    print(f"Change: {change.comment}")

# Executive summary
print(f"Summary: {result.executive_summary.overall_assessment}")
```

## 🛠️ Customization

### Custom Risk Patterns
```python
# Add custom risk detection patterns
from src.analysis.risk_assessor import RiskAssessor

risk_assessor = RiskAssessor()
risk_assessor.risk_patterns['CUSTOM_RISK'] = {
    'patterns': [r'\b(custom\s+pattern)\b'],
    'weight': 1.0,
    'keywords': ['custom', 'keyword']
}
```

### Custom Document Types
```python
# Extend document type classification
from src.core.models import DocumentType

# Add new document type to enum
# Then update classification patterns in legal_analyzer.py
```

### Custom LLM Prompts
```python
# Modify prompts in analysis modules
# Example: src/analysis/legal_analyzer.py
def _build_custom_prompt(self, content):
    return f"""
    Custom legal analysis prompt for: {content}
    
    Focus on specific legal areas...
    """
```

## ⚠️ Important Notes

### Legal Disclaimers
- **AI Assistance Only**: This tool assists legal professionals but does not replace human legal judgment
- **Review Required**: All AI analysis should be reviewed by qualified attorneys
- **No Legal Advice**: The tool does not provide legal advice
- **Professional Responsibility**: Users are responsible for compliance with legal ethics rules

### Security Considerations
- **Confidential Information**: Handle client documents securely
- **API Keys**: Protect API keys and credentials
- **Data Retention**: Implement appropriate data retention policies
- **Access Control**: Restrict access to authorized personnel only

### Performance Optimization
- **File Size**: Optimize document size for faster processing
- **Concurrent Requests**: Limit concurrent analyses based on API rate limits
- **Caching**: Enable caching for repeated analyses
- **Batch Processing**: Use batch processing for multiple documents

## 🐛 Troubleshooting

### Common Issues

#### "Import could not be resolved"
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path and virtual environment

#### "API key not configured"
- Add API keys to `.env` file
- Verify environment variable names match configuration

#### "File not found" errors
- Check file paths are absolute
- Ensure upload directory exists and is writable

#### "Analysis timeout"
- Increase timeout settings in configuration
- Check API rate limits and network connectivity

#### Low confidence scores
- Verify document quality and format
- Check for proper document structure
- Ensure appropriate document type classification

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in environment
LOG_LEVEL=DEBUG
```

### Performance Monitoring
```python
# Check processing times
result = await agent.review_document("contract.pdf")
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Confidence: {result.confidence_score:.2%}")
```

## 🤝 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the example code in `example.py`
3. Run tests with `python tests/test_agent.py`
4. Check configuration with health endpoint: `GET /health`

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.
