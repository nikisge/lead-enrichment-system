"""
Unified LLM Client for Lead Enrichment System.

Provides access to multiple LLM providers via OpenRouter for:
- Fast validations (Gemini 3 Flash Preview)
- Balanced extraction (Claude Haiku 4.5)
- Complex analysis (Claude Sonnet 4.5)

Cost-optimized: Uses cheapest model that can handle the task.
"""

import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelTier(Enum):
    """Model tiers for different use cases."""
    FAST = "fast"           # Cheapest, for simple validations
    BALANCED = "balanced"   # Good balance for extractions
    SMART = "smart"         # Best quality for complex tasks


# Model configuration - January 2026 (OpenRouter model IDs)
MODEL_CONFIG = {
    ModelTier.FAST: {
        "model": "google/gemini-3-flash-preview",  # Newest Gemini, fast & cheap
        "max_tokens": 1000,
        "temperature": 0.1,
        "cost_per_1m_input": 0.50,
        "cost_per_1m_output": 3.00,
    },
    ModelTier.BALANCED: {
        "model": "anthropic/claude-haiku-4.5",  # Claude 4.5 Haiku - best balance
        "max_tokens": 2000,
        "temperature": 0.1,
        "cost_per_1m_input": 0.80,
        "cost_per_1m_output": 4.00,
    },
    ModelTier.SMART: {
        "model": "anthropic/claude-sonnet-4.5",  # Claude 4.5 Sonnet - best quality
        "max_tokens": 4000,
        "temperature": 0.2,
        "cost_per_1m_input": 3.00,
        "cost_per_1m_output": 15.00,
    },
}

# Fallback models if primary not available
FALLBACK_MODELS = {
    "google/gemini-3-flash-preview": "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5": "anthropic/claude-3.5-haiku",
    "anthropic/claude-sonnet-4.5": "anthropic/claude-3.5-sonnet",
}


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    success: bool
    error: Optional[str] = None


class LLMClient:
    """
    Unified LLM client using OpenRouter.

    Provides easy access to different model tiers:
    - fast: Quick validations, simple yes/no decisions
    - balanced: Data extraction, moderate complexity
    - smart: Complex analysis, sales briefs

    Usage:
        llm = LLMClient()
        response = await llm.call("Is 'Max Müller' a real name?", tier="fast")

        # Or with JSON output
        data = await llm.call_json(prompt, tier="balanced")
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.openrouter_api_key
        self.anthropic_key = settings.anthropic_api_key  # Fallback
        self.timeout = settings.api_timeout
        self._total_cost = 0.0
        self._call_count = 0

    async def call(
        self,
        prompt: str,
        tier: Union[ModelTier, str] = ModelTier.FAST,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Make an LLM call with the specified tier.

        Args:
            prompt: User prompt
            tier: Model tier (fast, balanced, smart) or ModelTier enum
            system_prompt: Optional system prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            LLMResponse with content and metadata
        """
        # Convert string tier to enum
        if isinstance(tier, str):
            tier = ModelTier(tier)

        config = MODEL_CONFIG[tier]
        model = config["model"]

        # Use overrides or defaults
        actual_max_tokens = max_tokens or config["max_tokens"]
        actual_temperature = temperature if temperature is not None else config["temperature"]

        # Try OpenRouter first, fallback to direct Anthropic
        if self.api_key:
            response = await self._call_openrouter(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=actual_max_tokens,
                temperature=actual_temperature,
                config=config
            )
        elif self.anthropic_key and "anthropic" in model:
            # Fallback to direct Anthropic API
            response = await self._call_anthropic_direct(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=actual_max_tokens,
                temperature=actual_temperature,
                config=config
            )
        else:
            return LLMResponse(
                content="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_estimate=0,
                success=False,
                error="No API key configured (OPENROUTER_API_KEY or ANTHROPIC_API_KEY)"
            )

        # Track costs
        if response.success:
            self._total_cost += response.cost_estimate
            self._call_count += 1

        return response

    async def call_json(
        self,
        prompt: str,
        tier: Union[ModelTier, str] = ModelTier.FAST,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Union[Dict, List]]:
        """
        Make an LLM call and parse JSON response.

        Returns parsed JSON or None if parsing fails.
        """
        # Add JSON instruction to prompt if not present
        if "json" not in prompt.lower():
            prompt = prompt + "\n\nAntworte NUR mit validem JSON, keine anderen Texte."

        response = await self.call(
            prompt=prompt,
            tier=tier,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1  # Lower temperature for JSON
        )

        if not response.success:
            logger.warning(f"LLM call failed: {response.error}")
            return None

        return self._parse_json_response(response.content)

    async def _call_openrouter(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        config: dict
    ) -> LLMResponse:
        """Call OpenRouter API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{OPENROUTER_BASE_URL}/chat/completions"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://lead-enrichment.local",
                "X-Title": "Lead Enrichment System"
            }

            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            try:
                response = await client.post(url, json=body, headers=headers)

                # Handle rate limits with retry
                if response.status_code == 429:
                    logger.warning("OpenRouter rate limit, waiting 2s...")
                    await asyncio.sleep(2)
                    response = await client.post(url, json=body, headers=headers)

                response.raise_for_status()
                data = response.json()

                # Extract response
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                # Calculate cost
                cost = (
                    (input_tokens / 1_000_000) * config["cost_per_1m_input"] +
                    (output_tokens / 1_000_000) * config["cost_per_1m_output"]
                )

                logger.debug(f"LLM call: {model}, tokens: {input_tokens}+{output_tokens}, cost: ${cost:.4f}")

                return LLMResponse(
                    content=content,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_estimate=cost,
                    success=True
                )

            except httpx.HTTPStatusError as e:
                error_msg = f"OpenRouter error: {e.response.status_code}"
                logger.error(f"{error_msg} - {e.response.text}")

                # Try fallback model
                fallback = FALLBACK_MODELS.get(model)
                if fallback and fallback != model:
                    logger.info(f"Trying fallback model: {fallback}")
                    return await self._call_openrouter(
                        prompt, fallback, system_prompt, max_tokens, temperature, config
                    )

                return LLMResponse(
                    content="",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost_estimate=0,
                    success=False,
                    error=error_msg
                )

            except Exception as e:
                logger.error(f"OpenRouter call failed: {e}")
                return LLMResponse(
                    content="",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    cost_estimate=0,
                    success=False,
                    error=str(e)
                )

    async def _call_anthropic_direct(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        config: dict
    ) -> LLMResponse:
        """Fallback: Call Anthropic API directly."""
        try:
            from anthropic import AsyncAnthropic

            # Map OpenRouter model name to Anthropic model name
            anthropic_model = model.replace("anthropic/", "")
            if anthropic_model == "claude-haiku-4.5":
                anthropic_model = "claude-3-5-haiku-20241022"  # Latest Haiku
            elif anthropic_model == "claude-sonnet-4.5":
                anthropic_model = "claude-sonnet-4-20250514"  # Latest Sonnet

            client = AsyncAnthropic(api_key=self.anthropic_key)

            kwargs = {
                "model": anthropic_model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = await client.messages.create(**kwargs)

            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            cost = (
                (input_tokens / 1_000_000) * config["cost_per_1m_input"] +
                (output_tokens / 1_000_000) * config["cost_per_1m_output"]
            )

            return LLMResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_estimate=cost,
                success=True
            )

        except Exception as e:
            logger.error(f"Anthropic direct call failed: {e}")
            return LLMResponse(
                content="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_estimate=0,
                success=False,
                error=str(e)
            )

    def _parse_json_response(self, content: str) -> Optional[Union[Dict, List]]:
        """Parse JSON from LLM response, handling common issues."""
        if not content:
            return None

        # Clean the content
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Try to find JSON in the response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object or array using balanced bracket matching
        import re

        # Find the first [ or { and extract balanced JSON
        def extract_balanced_json(text: str, start_char: str, end_char: str) -> Optional[str]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                return None

            count = 0
            in_string = False
            escape = False
            end_idx = start_idx

            for i, char in enumerate(text[start_idx:], start_idx):
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if char == start_char:
                    count += 1
                elif char == end_char:
                    count -= 1
                    if count == 0:
                        end_idx = i + 1
                        break

            if count == 0 and end_idx > start_idx:
                return text[start_idx:end_idx]
            return None

        # Try array first (more common for contact extraction)
        json_str = extract_balanced_json(content, '[', ']')
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Array JSON parse failed: {e}")

        # Try object
        json_str = extract_balanced_json(content, '{', '}')
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Object JSON parse failed: {e}")

        # Fallback: try simple regex (non-greedy for cleaner extraction)
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from LLM response: {content[:300]}...")
        return None

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_cost": round(self._total_cost, 4),
            "average_cost_per_call": round(self._total_cost / max(1, self._call_count), 4)
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self._total_cost = 0.0
        self._call_count = 0


# Convenience function for quick access
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


async def quick_llm_call(
    prompt: str,
    tier: str = "fast",
    system_prompt: Optional[str] = None
) -> str:
    """Quick LLM call - returns content string or empty string on failure."""
    client = get_llm_client()
    response = await client.call(prompt, tier=tier, system_prompt=system_prompt)
    return response.content if response.success else ""


async def quick_llm_json(
    prompt: str,
    tier: str = "fast",
    system_prompt: Optional[str] = None
) -> Optional[Union[Dict, List]]:
    """Quick LLM call with JSON parsing."""
    client = get_llm_client()
    return await client.call_json(prompt, tier=tier, system_prompt=system_prompt)
