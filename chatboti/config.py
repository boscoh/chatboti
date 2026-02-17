"""Configuration and client initialization for chatboti.

Client API
----------

Chat clients returned by `get_chat_client()` implement the SimpleLLMClient interface:
    - model: str - Model name
    - async connect() -> None - Initialize connection
    - async close() -> None - Close connection
    - async chat(messages, tools=None, **kwargs) -> response - Send chat request
    - async stream_chat(messages, tools=None, **kwargs) -> AsyncIterator - Stream chat

Embed clients returned by `get_embed_client()` implement the SimpleLLMClient interface:
    - model: str - Model name
    - async connect() -> None - Initialize connection
    - async close() -> None - Close connection
    - async embed(texts: list[str]) -> list[list[float]] - Generate embeddings

Usage Example
-------------
    # Chat client
    chat_client = await get_chat_client()
    response = await chat_client.chat([{"role": "user", "content": "Hello"}])
    await chat_client.close()

    # Embed client
    embed_client = await get_embed_client()
    embeddings = await embed_client.embed(["text1", "text2"])
    await embed_client.close()

    # Or use as context manager
    async with await get_chat_client() as client:
        response = await client.chat(messages)
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from chatboti.llm import SimpleLLMClient, get_llm_client, load_config

logger = logging.getLogger(__name__)

# Track if env has been loaded to make load_env idempotent
_env_loaded = False


def load_env() -> bool:
    """Load environment variables from .env file (idempotent).

    Searches for .env file in:
    1. Current working directory
    2. Module parent directory (chatboti package location)

    :return: True if .env file was found and loaded (or already loaded)
    """
    global _env_loaded

    # Skip if already loaded (unless forced)
    if _env_loaded:
        return True

    # Try current working directory first
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        logger.info(f"Loading .env from: {cwd_env}")
        load_dotenv(cwd_env, verbose=True)
        _env_loaded = True
        return True

    # Try module parent directory
    module_dir = Path(__file__).parent.parent
    module_env = module_dir / ".env"
    if module_env.exists():
        logger.info(f"Loading .env from: {module_env}")
        load_dotenv(module_env, verbose=True)
        _env_loaded = True
        return True

    return False


@lru_cache()
def _get_default_model(service: str, model_type: str) -> str:
    """Get the default model for a service.

    :param model_type: 'chat_models' or 'embed_models'
    :param service: Service name
    :return: Default model name or empty string
    """
    config = load_config()
    models_dict = config.get(model_type, {})
    models = models_dict.get(service, [])
    if isinstance(models, list) and models:
        return models[0]
    if isinstance(models, str):
        return models
    return ""


def get_chat_service() -> str:
    """Get the chat service from environment variable or config.

    :return: Chat service name
    """
    service = os.getenv("CHAT_SERVICE")
    if not service:
        raise ValueError("CHAT_SERVICE environment variable is not set")
    return service


async def get_chat_client() -> SimpleLLMClient:
    """Create and connect a chat client with logging.

    :return: Connected chat client
    :raises ValueError: If service or model cannot be determined
    """
    load_env()
    service = get_chat_service()
    model = os.getenv("CHAT_MODEL") or _get_default_model(service, "chat_models")

    if not model:
        raise ValueError(f"No model configured for chat service '{service}'")

    logger.info(f"Chat client: {service}:{model}")
    logger.info(f"Connecting to {service}...")
    client = get_llm_client(service, model=model)
    await client.connect()
    logger.info(f"Connected to {service}:{model}")
    return client


def get_embed_service() -> str:
    """Get the embed service from environment variable or config.

    :return: Embed service name
    """
    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    if not service:
        raise ValueError(
            "Neither EMBED_SERVICE nor CHAT_SERVICE environment variable is set"
        )
    return service


async def get_embed_client() -> SimpleLLMClient:
    """Create and connect an embedding client with logging.

    :return: Connected embedding client
    :raises ValueError: If service cannot be determined
    """
    load_env()
    service = get_embed_service()
    model = os.getenv("EMBED_MODEL") or _get_default_model(service, "embed_models")

    logger.info(f"Embed client: {service}:{model}")
    logger.info(f"Connecting to {service}...")
    client = get_llm_client(service, model=model)
    await client.connect()
    logger.info(f"Connected to {service}:{model}")
    return client
