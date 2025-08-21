"""
LLM client for interacting with Claude (Anthropic) exclusively.
"""

import asyncio
from typing import Optional, Dict, Any, List
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_config


class LLMClient:
    """
    Client for interacting with Claude (Anthropic) exclusively.
    
    Handles:
    - Claude API communication
    - Rate limiting and retries
    - Token counting and optimization
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize the LLM client."""
        self.config = get_config()
        self.anthropic_client = None
        
        # Initialize Claude client
        if self.config.llm.has_anthropic_key:
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=self.config.llm.anthropic_api_key
            )
        else:
            raise ValueError("Claude API key is required but not found in configuration")
        
        # Claude model mappings
        self.anthropic_models = [
            "claude-sonnet-4-20250514",     # Latest Claude Sonnet 4
            "claude-3-sonnet-20240229",     # Claude 3 Sonnet
            "claude-3-haiku-20240307"       # Claude 3 Haiku
        ]
        
        self.default_model = "claude-sonnet-4-20250514"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The input prompt
            model: Specific Claude model to use (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for context
            
        Returns:
            Generated response text
        """
        model = model or self.default_model
        max_tokens = max_tokens or 20000
        temperature = temperature or 0.1
        
        try:
            return await self._generate_anthropic_response(
                prompt, model, max_tokens, temperature, system_prompt
            )
        except Exception as e:
            print(f"Error with Claude model {model}: {str(e)}")
            raise e

    async def _generate_anthropic_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response using Anthropic Claude API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Create request parameters
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Make the API call
            response = await self.anthropic_client.messages.create(**request_params)
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                return "No response generated"
                
        except Exception as e:
            print(f"Anthropic API error: {str(e)}")
            raise e

    async def analyze_document_clause(
        self,
        clause_text: str,
        document_type: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a specific clause in the document.
        
        Args:
            clause_text: The clause text to analyze
            document_type: Type of document being analyzed
            system_prompt: Optional system prompt
            
        Returns:
            Analysis results as a dictionary
        """
        prompt = f"""
Analyze this legal clause from a {document_type}:

CLAUSE TEXT:
{clause_text}

Provide analysis in JSON format with these fields:
- enforceability_score: Number from 1-10 rating enforceability
- issues: Array of specific issues found
- suggestions: Array of improvement suggestions
- risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
- clause_type: Type/category of this clause

Focus on legal accuracy, enforceability, and business risk.
"""
        
        try:
            response = await self.generate_response(
                prompt, 
                system_prompt=system_prompt or "You are an expert legal analyst reviewing contract clauses."
            )
            
            # Try to parse as JSON, fallback to structured text
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "enforceability_score": 7.0,
                    "issues": ["Analysis parsing failed"],
                    "suggestions": ["Manual review recommended"],
                    "risk_level": "MEDIUM",
                    "clause_type": "Unknown",
                    "raw_response": response
                }
                
        except Exception as e:
            print(f"Clause analysis failed: {str(e)}")
            return {
                "enforceability_score": 5.0,
                "issues": [f"Analysis failed: {str(e)}"],
                "suggestions": ["Manual review required"],
                "risk_level": "MEDIUM",
                "clause_type": "Unknown"
            }

    async def assess_document_risks(
        self,
        document_content: str,
        document_type: str,
        priority_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assess risks in the entire document.
        
        Args:
            document_content: Full document content
            document_type: Type of document
            priority_areas: Areas to focus risk assessment on
            
        Returns:
            Risk assessment results
        """
        priority_text = ""
        if priority_areas:
            priority_text = f"Focus particularly on these areas: {', '.join(priority_areas)}"
        
        prompt = f"""
Analyze this {document_type} for legal and business risks:

{priority_text}

DOCUMENT CONTENT:
{document_content[:3000]}...

Identify and categorize risks in JSON format with:
- risks: Array of risk objects with fields:
  - category: Risk category (Financial, Legal, Operational, etc.)
  - description: Clear description of the risk
  - severity_score: 1-10 severity rating
  - level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
  - impact: Business impact description
  - recommendations: Array of mitigation suggestions

Focus on actionable risks that impact business operations.
"""
        
        try:
            response = await self.generate_response(
                prompt,
                system_prompt="You are a senior legal risk analyst specializing in contract review."
            )
            
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "risks": [{
                        "category": "Analysis",
                        "description": "Risk assessment parsing failed",
                        "severity_score": 5,
                        "level": "MEDIUM",
                        "impact": "Manual review required",
                        "recommendations": ["Review document manually"]
                    }],
                    "raw_response": response
                }
                
        except Exception as e:
            print(f"Risk assessment failed: {str(e)}")
            return {
                "risks": [{
                    "category": "Analysis",
                    "description": f"Risk assessment failed: {str(e)}",
                    "severity_score": 5,
                    "level": "MEDIUM",
                    "impact": "Unable to assess risks automatically",
                    "recommendations": ["Manual legal review recommended"]
                }]
            }

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for Claude
        return len(text) // 4

    def truncate_to_token_limit(self, text: str, max_tokens: int = 3000) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        estimated_tokens = self.count_tokens(text)
        if estimated_tokens <= max_tokens:
            return text
        
        # Truncate to approximately the right number of characters
        target_chars = max_tokens * 4
        return text[:target_chars] + "..."
