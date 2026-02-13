#!/usr/bin/env python3

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from microeval.llm import load_config

from chatboti.generic_rag import GenericRAGService
from chatboti.logger import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()


def get_default_model(models_dict: dict, service: str) -> str:
    """Get the default model for a service (first in list or string value)."""
    models = models_dict.get(service, [])
    if isinstance(models, list) and models:
        return models[0]
    elif isinstance(models, str):
        return models
    return ""


# Load model configuration
model_config = load_config()
embed_models = model_config["embed_models"]

chat_service = os.getenv("CHAT_SERVICE")
if not chat_service:
    raise ValueError("CHAT_SERVICE environment variable is not set")
embed_service = os.getenv("EMBED_SERVICE") or chat_service

# Get model
model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, embed_service)

# Set up data directory
data_dir = Path(__file__).parent / "data"

# Create RAG service (will be initialized in lifespan)
rag_service = None


@asynccontextmanager
async def lifespan(app):
    global rag_service
    try:
        logger.info(f"Initializing RAG service with embed_service: {embed_service}")

        # Create RAG service using constructor and context manager
        rag_service = GenericRAGService(
            service_name=embed_service,
            model=model,
            data_dir=data_dir
        )
        await rag_service.__aenter__()
        logger.info(f"RAG service initialized: {len(rag_service.documents)} documents, {rag_service.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
        raise
    yield
    try:
        # Close RAG service (which closes the embed client)
        if rag_service:
            await rag_service.__aexit__(None, None, None)
        logger.info("RAG service closed successfully")
    except Exception as e:
        logger.warning(f"Error during RAG service cleanup: {e}")


mcp = FastMCP("Simple MCP", lifespan=lifespan)


@mcp.tool()
async def get_matching_speakers(query: str, n: int = 3) -> Dict[str, Any]:
    """
    Find the top N most relevant speakers for a given topic using AI-powered semantic search.

    Use this tool when you need to find speakers who can talk about a specific topic,
    technology, or subject area. The tool analyzes speaker bios and abstracts to find
    the best semantic matches for your query.

    Examples of good queries:
    - "machine learning and AI"
    - "cloud computing and DevOps"
    - "data science and analytics"
    - "software architecture and design patterns"
    - "cybersecurity and privacy"

    Args:
        query: A description of the topic, technology, or expertise area you need speakers for
        n: Number of top matching speakers to return (default: 3, max recommended: 5)

    Returns:
        Dict containing the top N matching speakers with their bios, abstracts, and relevance details
    """
    try:
        # Use generic RAG search to find top N matching documents
        results = await rag_service.search(query, k=n, include_documents=True)

        if not results:
            return {
                "success": False,
                "error": "No matching speakers found",
                "query": query,
                "n": n
            }

        # Extract speaker info from all results
        speakers = []
        for result in results:
            speaker_content = result.content or {}
            speakers.append(speaker_content)

        return {
            "success": True,
            "speakers": speakers,
            "count": len(speakers),
            "query": query,
            "total_speakers_searched": len(rag_service.documents),
        }
    except Exception as e:
        logger.error(f"Error in get_matching_speakers: {e}", exc_info=True)
        return {"success": False, "error": str(e), "query": query, "n": n}


@mcp.tool()
async def list_all_speakers() -> Dict[str, Any]:
    """
    Get a complete list of all available speaker names.

    Use this tool when you want to see the names of the available speakers.

    Returns:
        Dict containing a list of all speaker names
    """
    try:
        # Extract speaker names from all documents
        speaker_names = []
        for doc in rag_service.documents.values():
            if isinstance(doc.content, dict):
                name = doc.content.get("Name", "")
                if name:
                    speaker_names.append(name)

        return {
            "success": True,
            "speakers": speaker_names,
            "intro_message": "**Conference Speakers from the data**",
        }
    except Exception as e:
        logger.error(f"Error in list_all_speakers: {e}", exc_info=True)
        return {"success": False, "error": str(e), "speakers": []}


def main():
    """Main function to run the MCP server."""
    try:
        setup_logging()
        logger.info("Starting MCP Server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
