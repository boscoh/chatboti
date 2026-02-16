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

from microeval.llm import SimpleLLMClient, get_llm_client, load_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model_config():
    """Load and cache model configuration.

    :return: Model configuration dict
    """
    return load_config()


def _get_default_model(models_dict: dict, service: str) -> str:
    """Get the default model for a service.

    :param models_dict: Dictionary mapping service names to models
    :param service: Service name
    :return: Default model name or empty string
    """
    models = models_dict.get(service, [])
    if isinstance(models, list) and models:
        return models[0]
    if isinstance(models, str):
        return models
    return ""


def get_chat_config() -> tuple[str, str]:
    """Get chat service and model from environment and config.

    :return: Tuple of (service, model)
    :raises ValueError: If service or model cannot be determined
    """
    config = _get_model_config()

    service = os.getenv("CHAT_SERVICE")
    if not service:
        raise ValueError("CHAT_SERVICE environment variable is not set")

    model = (
        os.getenv("CHAT_MODEL")
        or os.getenv(f"{service.upper()}_MODEL")
        or _get_default_model(config["chat_models"], service)
    )

    if not model:
        raise ValueError(f"No model configured for chat service '{service}'")

    return service, model


def get_embed_config() -> tuple[str, str]:
    """Get embedding service and model from environment and config.

    :return: Tuple of (service, model)
    :raises ValueError: If service cannot be determined
    """
    config = _get_model_config()

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    if not service:
        raise ValueError("Neither EMBED_SERVICE nor CHAT_SERVICE environment variable is set")

    model = os.getenv("EMBED_MODEL") or _get_default_model(config["embed_models"], service)

    return service, model


async def _create_client(service: str, model: str) -> SimpleLLMClient:
    """Create and connect a client.

    :param service: Service name
    :param model: Model name
    :return: Connected client
    """
    client = get_llm_client(service, model=model)
    await client.connect()
    return client


async def get_chat_client() -> SimpleLLMClient:
    """Create and connect a chat client with logging.

    :return: Connected chat client
    :raises ValueError: If service or model cannot be determined
    """
    service, model = get_chat_config()
    logger.info(f"Chat client: {service}:{model}")
    logger.info(f"Connecting to {service}...")
    client = await _create_client(service, model)
    logger.info(f"Connected to {service}:{model}")
    return client


async def get_embed_client() -> SimpleLLMClient:
    """Create and connect an embedding client with logging.

    :return: Connected embedding client
    :raises ValueError: If service cannot be determined
    """
    service, model = get_embed_config()
    logger.info(f"Embed client: {service}:{model}")
    logger.info(f"Connecting to {service}...")
    client = await _create_client(service, model)
    logger.info(f"Connected to {service}:{model}")
    return client
