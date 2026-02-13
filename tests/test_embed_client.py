"""Tests for embedding client wrappers."""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from chatboti.embed_client import (
    EmbedClient,
    OpenAIEmbedClient,
    OllamaEmbedClient,
    MicroevalEmbedClient,
)


class ConcreteEmbedClient(EmbedClient):
    """Concrete implementation for testing abstract base class."""

    async def embed(self, text: str):
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
class TestEmbedClient:
    """Test the abstract EmbedClient base class."""

    async def test_concrete_implementation(self):
        """Test that concrete implementation works."""
        client = ConcreteEmbedClient(model="test-model")
        assert client.model == "test-model"

        embedding = await client.embed("test text")
        assert embedding == [0.1, 0.2, 0.3]

    async def test_default_connect_close(self):
        """Test default connect/close do nothing."""
        client = ConcreteEmbedClient(model="test-model")
        await client.connect()
        await client.close()

    async def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            EmbedClient(model="test-model")


@pytest.mark.asyncio
class TestOpenAIEmbedClient:
    """Test OpenAI embedding client."""

    async def test_init_default_model(self):
        """Test initialization with default model."""
        client = OpenAIEmbedClient()
        assert client.model == "text-embedding-3-small"
        assert client.api_key is None

    async def test_init_custom_model_and_key(self):
        """Test initialization with custom model and API key."""
        client = OpenAIEmbedClient(model="custom-model", api_key="test-key")
        assert client.model == "custom-model"
        assert client.api_key == "test-key"

    async def test_connect(self):
        """Test connection initializes OpenAI client."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            client = OpenAIEmbedClient(api_key="test-key")
            await client.connect()

            mock_openai.assert_called_once_with(api_key="test-key")
            assert client._client is not None

    async def test_embed(self):
        """Test embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            client = OpenAIEmbedClient()
            embedding = await client.embed("test text")

            assert embedding == [0.1, 0.2, 0.3]
            mock_client.embeddings.create.assert_called_once_with(
                input="test text",
                model="text-embedding-3-small"
            )

    async def test_close(self):
        """Test closing the client."""
        mock_client = AsyncMock()

        client = OpenAIEmbedClient()
        client._client = mock_client

        await client.close()

        mock_client.close.assert_called_once()
        assert client._client is None

    async def test_import_error(self):
        """Test ImportError when openai not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            client = OpenAIEmbedClient()

            with pytest.raises(ImportError, match="openai package not installed"):
                await client.connect()


@pytest.mark.asyncio
class TestOllamaEmbedClient:
    """Test Ollama embedding client."""

    async def test_init_default_model(self):
        """Test initialization with default model."""
        client = OllamaEmbedClient()
        assert client.model == "nomic-embed-text"
        assert client.host is None

    async def test_init_custom_model_and_host(self):
        """Test initialization with custom model and host."""
        client = OllamaEmbedClient(model="custom-model", host="http://localhost:11434")
        assert client.model == "custom-model"
        assert client.host == "http://localhost:11434"

    async def test_connect_default_host(self):
        """Test connection with default host."""
        with patch("ollama.AsyncClient") as mock_client_class:
            client = OllamaEmbedClient()
            await client.connect()

            mock_client_class.assert_called_once_with()
            assert client._client is not None

    async def test_connect_custom_host(self):
        """Test connection with custom host."""
        with patch("ollama.AsyncClient") as mock_client_class:
            client = OllamaEmbedClient(host="http://custom:11434")
            await client.connect()

            mock_client_class.assert_called_once_with(host="http://custom:11434")
            assert client._client is not None

    async def test_embed(self):
        """Test embedding generation."""
        mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=mock_response)

        with patch("ollama.AsyncClient", return_value=mock_client):
            client = OllamaEmbedClient()
            embedding = await client.embed("test text")

            assert embedding == [0.1, 0.2, 0.3]
            mock_client.embed.assert_called_once_with(
                model="nomic-embed-text",
                input="test text"
            )

    async def test_import_error(self):
        """Test ImportError when ollama not installed."""
        with patch.dict("sys.modules", {"ollama": None}):
            client = OllamaEmbedClient()

            with pytest.raises(ImportError, match="ollama package not installed"):
                await client.connect()


@pytest.mark.asyncio
class TestMicroevalEmbedClient:
    """Test microeval embedding client wrapper."""

    async def test_init(self):
        """Test initialization."""
        client = MicroevalEmbedClient(service="openai", model="text-embedding-3-small")
        assert client.service == "openai"
        assert client.model == "text-embedding-3-small"

    async def test_connect(self):
        """Test connection initializes microeval client."""
        mock_llm_client = AsyncMock()
        mock_llm_client.connect = AsyncMock()

        with patch("microeval.llm.get_llm_client", return_value=mock_llm_client):
            client = MicroevalEmbedClient(service="openai", model="test-model")
            await client.connect()

            assert client._client is not None
            mock_llm_client.connect.assert_called_once()

    async def test_embed(self):
        """Test embedding generation."""
        mock_llm_client = AsyncMock()
        mock_llm_client.connect = AsyncMock()
        mock_llm_client.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch("microeval.llm.get_llm_client", return_value=mock_llm_client):
            client = MicroevalEmbedClient(service="openai", model="test-model")
            embedding = await client.embed("test text")

            assert embedding == [0.1, 0.2, 0.3]
            mock_llm_client.embed.assert_called_once_with("test text")

    async def test_close(self):
        """Test closing the client."""
        mock_llm_client = AsyncMock()
        mock_llm_client.connect = AsyncMock()
        mock_llm_client.close = AsyncMock()

        with patch("microeval.llm.get_llm_client", return_value=mock_llm_client):
            client = MicroevalEmbedClient(service="openai", model="test-model")
            await client.connect()
            await client.close()

            mock_llm_client.close.assert_called_once()

    async def test_import_error(self):
        """Test ImportError when microeval not installed."""
        with patch.dict("sys.modules", {"microeval": None, "microeval.llm": None}):
            client = MicroevalEmbedClient(service="openai", model="test-model")

            with pytest.raises(ImportError, match="microeval package not installed"):
                await client.connect()


@pytest.mark.asyncio
class TestClientComparison:
    """Test that all clients have consistent interface."""

    async def test_all_clients_have_same_interface(self):
        """Test that all concrete clients implement the same interface."""
        clients = [
            OpenAIEmbedClient(),
            OllamaEmbedClient(),
            MicroevalEmbedClient(service="openai", model="test-model"),
        ]

        for client in clients:
            assert hasattr(client, "model")
            assert hasattr(client, "embed")
            assert hasattr(client, "connect")
            assert hasattr(client, "close")
