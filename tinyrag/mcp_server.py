#!/usr/bin/env python3

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tinyrag.rag import RAGService
from tinyrag.logger import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()
chat_service = os.getenv("CHAT_SERVICE")
if not chat_service:
    raise ValueError("CHAT_SERVICE environment variable is not set")
embed_service = os.getenv("EMBED_SERVICE") or chat_service
rag_service = RAGService(llm_service=embed_service)


@asynccontextmanager
async def lifespan(app):
    await rag_service.__aenter__()
    yield
    await rag_service.__aexit__(None, None, None)


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
        best_speaker = await rag_service.get_best_speaker(query)
        return {
            "success": True,
            "speaker": best_speaker,
            "query": query,
            "total_speakers_searched": len(rag_service.speakers_with_embeddings),
        }
    except Exception as e:
        logger.error(f"Error in get_best_speaker: {e}")
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
        speakers = await rag_service.get_speakers()
        return {
            "success": True,
            "speakers": [speaker.get("name") for speaker in speakers],
            "intro_message": "**Conference Speakers from the data**",
        }
    except Exception as e:
        logger.error(f"Error in list_all_speakers: {e}")
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
