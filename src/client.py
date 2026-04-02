"""LM Studio API client wrapper for LLM communication."""

import requests
from typing import Optional


class LLMClient:
    """Client for interacting with LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: int = 300,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: The base URL for the LM Studio API (default: localhost:1234)
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url
        self.completions_url = f"{base_url}/chat/completions"
        self.timeout = timeout

    def generate(
        self,
        model_id: str,
        system_msg: str,
        user_msg: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            model_id: The model identifier (as it appears in LM Studio)
            system_msg: The system prompt/instruction
            user_msg: The user message/content
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            The generated text response

        Raises:
            requests.RequestException: If the API call fails
        """
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(self.completions_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def generate_with_history(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response with a full conversation history.

        Args:
            model_id: The model identifier
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response
        """
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(self.completions_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def check_health(self) -> bool:
        """
        Check if the LM Studio API is accessible.

        Returns:
            True if the API is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
