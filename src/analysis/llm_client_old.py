"""
LLM client for interacting with Claude (Anthropic).
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
            "claude-3-sonnet-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    
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
        Generate a response using the specified or default LLM.
        
        Args:
            prompt: The input prompt
            model: Specific model to use (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: System prompt for context
            
        Returns:
            Generated response text
        """
        model = model or self.config.llm.default_llm
        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature
        
        try:
            if model.startswith("gpt") or model.startswith("o1"):
                return await self._generate_openai_response(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            elif model.startswith("claude"):
                return await self._generate_anthropic_response(
                    prompt, model, max_tokens, temperature, system_prompt
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")
            # Try fallback model
            return await self._generate_fallback_response(
                prompt, max_tokens, temperature, system_prompt
            )
    
    async def _generate_openai_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response using OpenAI API."""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    async def _generate_anthropic_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response using Anthropic API."""
        if not self.anthropic_client:
            raise Exception("Anthropic client not initialized")
        
        # Combine system prompt with user prompt for Anthropic
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text
    
    async def _generate_fallback_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response using fallback model."""
        # Try available models in order of preference
        if self.anthropic_client:
            for model in self.anthropic_models:
                try:
                    return await self._generate_anthropic_response(
                        prompt, model, max_tokens, temperature, system_prompt
                    )
                except Exception as e:
                    print(f"Fallback failed with {model}: {str(e)}")
                    continue
        
        if self.openai_client:
            for model in self.openai_models:
                try:
                    return await self._generate_openai_response(
                        prompt, model, max_tokens, temperature, system_prompt
                    )
                except Exception as e:
                    print(f"Fallback failed with {model}: {str(e)}")
                    continue
        
        raise Exception("All LLM providers failed")
    
    async def generate_structured_response(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response that conforms to a given schema.
        
        Args:
            prompt: The input prompt
            schema: JSON schema for the expected response
            model: Specific model to use
            
        Returns:
            Structured response as dictionary
        """
        # Add schema instructions to prompt
        structured_prompt = f"""
{prompt}

Please respond with a JSON object that follows this schema:
{schema}

Ensure your response is valid JSON and includes all required fields.
"""
        
        response = await self.generate_response(
            structured_prompt,
            model=model,
            temperature=0.1  # Lower temperature for structured output
        )
        
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return error structure
            return {"error": "Failed to parse structured response", "raw_response": response}
    
    async def analyze_document_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[str]:
        """
        Analyze multiple prompts concurrently with rate limiting.
        
        Args:
            prompts: List of prompts to process
            model: Model to use for all prompts
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses in same order as prompts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_prompt(prompt: str) -> str:
            async with semaphore:
                return await self.generate_response(prompt, model=model)
        
        tasks = [process_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated text
        """
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate approximate character limit
        char_limit = max_tokens * 4
        
        if len(text) <= char_limit:
            return text
        
        # Truncate and add ellipsis
        return text[:char_limit-3] + "..."
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models for each provider.
        
        Returns:
            Dictionary mapping provider to available models
        """
        available = {}
        
        if self.openai_client:
            try:
                models = await self.openai_client.models.list()
                available["openai"] = [model.id for model in models.data]
            except Exception as e:
                print(f"Failed to get OpenAI models: {str(e)}")
                available["openai"] = self.openai_models
        
        if self.anthropic_client:
            # Anthropic doesn't have a models list endpoint
            available["anthropic"] = self.anthropic_models
        
        return available
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of all configured LLM providers.
        
        Returns:
            Dictionary mapping provider to health status
        """
        health = {}
        
        # Test OpenAI
        if self.openai_client:
            try:
                await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1
                )
                health["openai"] = True
            except Exception:
                health["openai"] = False
        else:
            health["openai"] = False
        
        # Test Anthropic
        if self.anthropic_client:
            try:
                await self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                health["anthropic"] = True
            except Exception:
                health["anthropic"] = False
        else:
            health["anthropic"] = False
        
        return health
