#!/usr/bin/env python3

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from microeval.llm import get_llm_client, load_config

from chatboti.generic_rag import GenericRAGService
from chatboti.logger import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()

# Load model configuration
model_config = load_config()
embed_models = model_config["embed_models"]

chat_service = os.getenv("CHAT_SERVICE")
if not chat_service:
    raise ValueError("CHAT_SERVICE environment variable is not set")
embed_service = os.getenv("EMBED_SERVICE") or chat_service

# Get model and create paths
model = os.getenv("EMBED_MODEL") or embed_models.get(embed_service)
if not model:
    raise ValueError(f"EMBED_MODEL not set for service {embed_service}")

# Import model slug function
from chatboti.rag_cli import make_model_slug
model_slug = make_model_slug(model)

# Set up paths
data_dir = Path(__file__).parent / "data"
index_path = data_dir / f"vectors-{model_slug}.faiss"
metadata_path = data_dir / f"metadata-{model_slug}.json"

# Load embedding dimension from metadata
if not metadata_path.exists():
    raise FileNotFoundError(
        f"RAG index not found at {metadata_path}. "
        f"Run 'chatboti build-rag' to generate embeddings first."
    )

with open(metadata_path) as f:
    metadata = json.load(f)
    embedding_dim = metadata.get('embedding_dim', 768)

# Create embed client
embed_client = get_llm_client(embed_service, model=model)

# Create RAG service (will be initialized in lifespan)
rag_service = None


@asynccontextmanager
async def lifespan(app):
    global rag_service
    try:
        logger.info(f"Initializing RAG service with embed_service: {embed_service}")

        # Connect embed client
        await embed_client.connect()
        logger.info(f"Embed client connected: {embed_service}:{model}")

        # Create and load RAG service
        rag_service = GenericRAGService(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=embedding_dim,
            embed_client=embed_client
        )
        logger.info(f"RAG service initialized: {len(rag_service.documents)} documents, {rag_service.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
        raise
    yield
    try:
        if embed_client:
            await embed_client.close()
        logger.info("RAG service closed successfully")
    except Exception as e:
        logger.warning(f"Error during RAG service cleanup: {e}")


mcp = FastMCP("Simple MCP", lifespan=lifespan)


@mcp.tool()
async def get_best_speaker(query: str) -> Dict[str, Any]:
    """
    Find the most relevant speaker for a given topic using AI-powered semantic search.

    Use this tool when you need to find a speaker who can talk about a specific topic,
    technology, or subject area. The tool analyzes speaker bios and abstracts to find
    the best semantic match for your query.

    Examples of good queries:
    - "machine learning and AI"
    - "cloud computing and DevOps"
    - "data science and analytics"
    - "software architecture and design patterns"
    - "cybersecurity and privacy"

    Args:
        query: A description of the topic, technology, or expertise area you need a speaker for

    Returns:
        Dict containing the best matching speaker with their bio, abstract, and relevance details
    """
    try:
        # Use generic RAG search to find best matching document
        results = await rag_service.search(query, k=1, include_documents=True)

        if not results:
            return {
                "success": False,
                "error": "No matching speaker found",
                "query": query
            }

        # Extract speaker info from the top result's document
        top_result = results[0]
        speaker_doc = top_result.document
        speaker_content = speaker_doc.content if isinstance(speaker_doc.content, dict) else {}

        # Format speaker data to match old API
        speaker = {
            "name": speaker_content.get("Name", ""),
            "bio_max_120_words": speaker_content.get("Bio (Max. 120 words)", ""),
            "final_abstract_max_150_words": speaker_content.get("Final abstract (Max. 150 words)", ""),
            "role": speaker_content.get("Role", ""),
            "country": speaker_content.get("Country", ""),
            "final_title": speaker_content.get("Final title", ""),
        }

        return {
            "success": True,
            "speaker": speaker,
            "query": query,
            "total_speakers_searched": len(rag_service.documents),
        }
    except Exception as e:
        logger.error(f"Error in get_best_speaker: {e}", exc_info=True)
        return {"success": False, "error": str(e), "query": query}


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
