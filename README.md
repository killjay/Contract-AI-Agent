# Legal Document Review AI Agent

A comprehensive AI-powered legal document analysis and review system that provides automated contract review, risk assessment, and redlining capabilities with the expertise of a senior attorney.

## 🎯 Features

- **Document Processing**: Parse PDF, Word, and text documents while preserving formatting
- **Legal Analysis**: Multi-phase analysis including structural, risk, and clause-by-clause review
- **Risk Assessment**: Comprehensive risk scoring with severity levels and mitigation strategies
- **Redlining Engine**: Generate tracked changes and detailed comments
- **Precedent Research**: Integration with legal databases and case law
- **Executive Summaries**: Business-friendly reports for stakeholders
- **API Interface**: RESTful API for integration with existing systems
- **Web Interface**: User-friendly Streamlit interface for document upload and review

## 🏗️ Architecture

The system is built with a modular architecture:

- **Core Agent**: Main orchestration engine
- **Document Parser**: Multi-format document processing
- **Legal Analyzer**: AI-powered legal analysis
- **Risk Assessor**: Comprehensive risk evaluation
- **Redlining Engine**: Change tracking and commenting
- **Precedent Database**: Legal research and case law
- **Workflow Manager**: Multi-step process orchestration

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key or Anthropic Claude API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the application:
   ```bash
   # Start the API server
   python -m src.api.main

   # Or start the Streamlit interface
   streamlit run src/ui/app.py
   ```

## 📋 Usage

### API Interface

```python
import requests

# Upload and analyze a document
response = requests.post(
    "http://localhost:8000/analyze",
    files={"file": open("contract.pdf", "rb")},
    data={"document_type": "employment_agreement"}
)

result = response.json()
```

### Python SDK

```python
from src.agent import LegalDocumentReviewAgent

agent = LegalDocumentReviewAgent()
result = await agent.review_document("path/to/contract.pdf")

print(f"Risk Score: {result.risk_assessment.overall_score}")
print(f"Critical Issues: {len(result.risk_assessment.critical_issues)}")
```

## 🔧 Configuration

The system can be configured through environment variables:

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models
- `DEFAULT_LLM`: Default language model to use
- `VECTOR_DB_PATH`: Path to vector database storage
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## 📊 Supported Document Types

- Employment Agreements
- Merger & Acquisition Agreements
- Commercial Contracts
- Non-Disclosure Agreements
- License Agreements
- Lease Agreements
- Service Agreements
- Partnership Agreements
- And more...

## ⚠️ Legal Disclaimer

This AI agent is designed to assist legal professionals and should not replace human legal judgment. All output should be reviewed by qualified attorneys before use in legal proceedings or decision-making.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📞 Support

For questions or support, please contact [your-email@example.com]
