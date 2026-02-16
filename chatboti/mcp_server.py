#!/usr/bin/env python3

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from microeval.llm import SimpleLLMClient

from chatboti.config import get_embed_client
from chatboti.faiss_rag import FaissRAGService
from chatboti.logger import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()

chat_service = os.getenv("CHAT_SERVICE")
if not chat_service:
    raise ValueError("CHAT_SERVICE environment variable is not set")

data_dir = Path(__file__).parent / "data"

rag_service: Optional[FaissRAGService] = None
embed_client: Optional[SimpleLLMClient] = None


@asynccontextmanager
async def lifespan(app):
    global rag_service, embed_client
    try:
        logger.info("Initializing embed client and RAG service...")

        embed_client = await get_embed_client()

        rag_service = FaissRAGService(
            embed_client=embed_client,
            data_dir=data_dir
        )
        await rag_service.__aenter__()
        logger.info(f"RAG service initialized: {len(rag_service.documents)} documents, {rag_service.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
        raise
    yield
    try:
        if rag_service:
            await rag_service.__aexit__(None, None, None)
        logger.info("RAG service closed successfully")

        if embed_client:
            await embed_client.close()
        logger.info("Embed client closed successfully")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


mcp = FastMCP("Simple MCP", lifespan=lifespan)


@mcp.tool()
async def get_matching_speakers(query: str, n: int = 3) -> Dict[str, Any]:
    """Find the top N most relevant speakers using semantic search.

    Use when you need speakers for a specific topic, technology, or subject area.

    Examples: "machine learning and AI", "cloud computing and DevOps"

    :param query: Topic, technology, or expertise area
    :param n: Number of results (default: 3, max recommended: 5)
    :return: Dict with matching speakers, bios, abstracts, and relevance details
    """
    try:
        results = await rag_service.search(query, k=n, include_documents=True)

        if not results:
            return {
                "success": False,
                "error": "No matching speakers found",
                "query": query,
                "n": n
            }

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
    """Get list of all available speaker names.

    :return: Dict containing speaker names
    """
    try:
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
    """Run the MCP server."""
    try:
        setup_logging()
        logger.info("Starting MCP Server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
