"""
Configuration management for the Legal Document Review AI Agent.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment variables."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    # Fall back to environment variables
    return os.getenv(key, default)


class LLMConfig(BaseModel):
    """Configuration for Language Model providers."""
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    default_llm: str = Field(default="claude-sonnet-4-20250514")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=20000, gt=0)
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key)
    
    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.anthropic_api_key)


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    db_type: str = Field(default="chromadb")
    db_path: str = Field(default="./data/vector_db")
    collection_name: str = Field(default="legal_documents")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


class APIConfig(BaseModel):
    """Configuration for API server."""
    host: str = Field(default="localhost")
    port: int = Field(default=8000, gt=0, le=65535)
    workers: int = Field(default=1, gt=0)
    secret_key: str = Field(default="change-this-secret-key")
    enable_auth: bool = Field(default=False)
    jwt_expiration_hours: int = Field(default=24, gt=0)


class StorageConfig(BaseModel):
    """Configuration for file storage."""
    upload_dir: str = Field(default="./data/uploads")
    output_dir: str = Field(default="./data/outputs")
    temp_dir: str = Field(default="./data/temp")
    max_file_size_mb: int = Field(default=50, gt=0)
    allowed_file_types: list = Field(default=["pdf", "docx", "doc", "txt"])


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    max_concurrent_reviews: int = Field(default=3, gt=0)
    document_chunk_size: int = Field(default=1000, gt=0)
    overlap_size: int = Field(default=200, ge=0)
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, gt=0)


class RiskConfig(BaseModel):
    """Configuration for risk assessment thresholds."""
    critical_threshold: int = Field(default=80, ge=0, le=100)
    high_threshold: int = Field(default=60, ge=0, le=100)
    medium_threshold: int = Field(default=40, ge=0, le=100)


class LegalDBConfig(BaseModel):
    """Configuration for legal database integrations."""
    westlaw_api_key: Optional[str] = Field(default=None)
    lexis_nexis_api_key: Optional[str] = Field(default=None)
    enable_google_scholar: bool = Field(default=True)
    enable_free_databases: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    legal_db: LegalDBConfig = Field(default_factory=LegalDBConfig)
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                openai_api_key=get_secret("OPENAI_API_KEY"),
                anthropic_api_key=get_secret("ANTHROPIC_API_KEY"),
                default_llm=get_secret("LLM_MODEL", "claude-sonnet-4-20250514"),
                temperature=float(get_secret("LLM_TEMPERATURE", "0.1")),
                max_tokens=int(get_secret("MAX_TOKENS", "20000"))
            ),
            vector_db=VectorDBConfig(
                db_type=os.getenv("VECTOR_DB_TYPE", "chromadb"),
                db_path=os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
                collection_name=os.getenv("VECTOR_COLLECTION", "legal_documents"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            api=APIConfig(
                host=os.getenv("API_HOST", "localhost"),
                port=int(os.getenv("API_PORT", "8000")),
                workers=int(os.getenv("API_WORKERS", "1")),
                secret_key=os.getenv("SECRET_KEY", "change-this-secret-key"),
                enable_auth=os.getenv("ENABLE_AUTH", "false").lower() == "true",
                jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
            ),
            storage=StorageConfig(
                upload_dir=os.getenv("UPLOAD_DIR", "./data/uploads"),
                output_dir=os.getenv("OUTPUT_DIR", "./data/outputs"),
                temp_dir=os.getenv("TEMP_DIR", "./data/temp"),
                max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
                allowed_file_types=os.getenv("ALLOWED_FILE_TYPES", "pdf,docx,doc,txt").split(",")
            ),
            performance=PerformanceConfig(
                max_concurrent_reviews=int(os.getenv("MAX_CONCURRENT_REVIEWS", "3")),
                document_chunk_size=int(os.getenv("DOCUMENT_CHUNK_SIZE", "1000")),
                overlap_size=int(os.getenv("OVERLAP_SIZE", "200"))
            ),
            risk=RiskConfig(
                critical_threshold=int(os.getenv("CRITICAL_RISK_THRESHOLD", "80")),
                high_threshold=int(os.getenv("HIGH_RISK_THRESHOLD", "60")),
                medium_threshold=int(os.getenv("MEDIUM_RISK_THRESHOLD", "40"))
            ),
            legal_db=LegalDBConfig(
                westlaw_api_key=os.getenv("WESTLAW_API_KEY"),
                lexis_nexis_api_key=os.getenv("LEXIS_NEXIS_API_KEY"),
                enable_google_scholar=os.getenv("ENABLE_GOOGLE_SCHOLAR", "true").lower() == "true",
                enable_free_databases=os.getenv("ENABLE_FREE_DATABASES", "true").lower() == "true"
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check LLM configuration
        if not self.llm.has_openai_key and not self.llm.has_anthropic_key:
            validation_results["errors"].append("No LLM API keys configured")
            validation_results["valid"] = False
        
        # Check storage directories
        for dir_path in [self.storage.upload_dir, self.storage.output_dir, self.storage.temp_dir]:
            if not os.path.exists(dir_path):
                validation_results["warnings"].append(f"Directory does not exist: {dir_path}")
        
        # Check risk thresholds
        if self.risk.critical_threshold <= self.risk.high_threshold:
            validation_results["warnings"].append("Critical risk threshold should be higher than high risk threshold")
        
        if self.risk.high_threshold <= self.risk.medium_threshold:
            validation_results["warnings"].append("High risk threshold should be higher than medium risk threshold")
        
        return validation_results


# Global configuration instance
config = Config.from_env()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> Config:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
