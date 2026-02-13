"""Embedding client wrappers for various LLM services.

This module provides an abstract interface for embedding clients and implementations
for OpenAI, Ollama, and microeval-based services.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbedClient(ABC):
    """Abstract base class for embedding clients.

    All embedding clients should inherit from this class and implement
    the embed() method. Optional connect() and close() methods can be
    overridden for clients that require connection management.
    """

    def __init__(self, model: str):
        """Initialize the embedding client.

        :param model: Model identifier for the embedding service
        """
        self.model = model

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text.

        :param text: Text to embed
        :return: Embedding vector as list of floats
        """
        pass

    async def connect(self):
        """Initialize connection to the embedding service.

        Override this method if your client requires connection setup.
        Default implementation does nothing.
        """
        pass

    async def close(self):
        """Close connection to the embedding service.

        Override this method if your client requires cleanup.
        Default implementation does nothing.
        """
        pass


class OpenAIEmbedClient(EmbedClient):
    """OpenAI embedding client using the openai library."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """Initialize OpenAI embedding client.

        :param model: OpenAI embedding model name
        :param api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        super().__init__(model)
        self.api_key = api_key
        self._client = None

    async def connect(self):
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self._client = AsyncOpenAI(api_key=self.api_key)

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API.

        :param text: Text to embed
        :return: Embedding vector as list of floats
        """
        if self._client is None:
            await self.connect()

        response = await self._client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    async def close(self):
        """Close the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class OllamaEmbedClient(EmbedClient):
    """Ollama local embedding client using the ollama library."""

    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None):
        """Initialize Ollama embedding client.

        :param model: Ollama model name
        :param host: Ollama host URL (if None, uses default localhost:11434)
        """
        super().__init__(model)
        self.host = host
        self._client = None

    async def connect(self):
        """Initialize the Ollama async client."""
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self._client = AsyncClient(host=self.host) if self.host else AsyncClient()

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama API.

        :param text: Text to embed
        :return: Embedding vector as list of floats
        """
        if self._client is None:
            await self.connect()

        response = await self._client.embed(model=self.model, input=text)
        return response["embeddings"][0]


class MicroevalEmbedClient(EmbedClient):
    """Wrapper for microeval llm client.

    This provides a unified interface to microeval's multi-service
    LLM client, supporting OpenAI, Ollama, Bedrock, and Groq.
    """

    def __init__(self, service: str, model: str):
        """Initialize microeval embedding client.

        :param service: Service name (openai, ollama, bedrock, groq)
        :param model: Model identifier for the service
        """
        super().__init__(model)
        self.service = service
        self._client = None

    async def connect(self):
        """Initialize the microeval client."""
        try:
            from microeval.llm import get_llm_client
        except ImportError:
            raise ImportError(
                "microeval package not installed. Install with: pip install microeval"
            )

        self._client = get_llm_client(self.service, model=self.model)
        await self._client.connect()

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using microeval client.

        :param text: Text to embed
        :return: Embedding vector as list of floats
        """
        if self._client is None:
            await self.connect()

        return await self._client.embed(text)

    async def close(self):
        """Close the microeval client."""
        if self._client is not None:
            await self._client.close()
