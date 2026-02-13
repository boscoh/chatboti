#!/usr/bin/env python3
"""
Chatboti CLI - Command-line interface for Chatboti
"""

import asyncio
import os
import sys

from cyclopts import App

from chatboti.server import run_server
from chatboti.agent import amain as agent_amain
from chatboti.rag import build_embeddings as rag_amain, search_loop
from chatboti.rag_cli import build_embeddings as build_rag_new, search_rag
from chatboti.docker import main as run_docker_main
from chatboti.logger import setup_logging

setup_logging()


app = App(name="chatboti", help="Chatboti - RAG starter kit")


@app.command(name="ui-chat", sort_key=0)
def ui_chat():
    """Start the UI with FastAPI backend and open browser."""
    run_server("127.0.0.1", 8000, open_browser=True, reload=False)


@app.command(sort_key=1)
def server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server only.

    :param host: Server host.
    :param port: Server port.
    :param reload: Enable auto-reload on file changes.
    """
    run_server(host, port, open_browser=False, reload=reload)


@app.command(name="cli-chat", sort_key=2)
def cli_chat():
    """Interactive chat with agent using MCP tools."""
    service = os.getenv("CHAT_SERVICE")
    if not service:
        print("Error: CHAT_SERVICE environment variable is not set")
        sys.exit(1)
    asyncio.run(agent_amain(service))


@app.command(sort_key=3)
def rag():
    """Generate embeddings for RAG."""
    asyncio.run(rag_amain())


@app.command(sort_key=4)
def search():
    """Interactive search loop to query the RAG database."""
    asyncio.run(search_loop())


@app.command(name="build-rag", sort_key=5)
def build_rag(
    csv_path: str = "",
    index_path: str = "",
    metadata_path: str = ""
):
    """Build RAG embeddings using GenericRAGService.

    All paths default to chatboti/data/ directory.

    :param csv_path: Path to CSV file (default: chatboti/data/2025-09-02-speaker-bio.csv)
    :param index_path: Path to save FAISS index (default: chatboti/data/vectors.faiss)
    :param metadata_path: Path to save metadata JSON (default: chatboti/data/metadata.json)
    """
    asyncio.run(build_rag_new(
        csv_path if csv_path else None,
        index_path if index_path else None,
        metadata_path if metadata_path else None
    ))


@app.command(name="search-rag", sort_key=6)
def search_rag_cmd(
    query: str,
    k: int = 5,
    index_path: str = "",
    metadata_path: str = ""
):
    """Search the RAG index.

    Paths default to chatboti/data/ directory.

    :param query: Search query
    :param k: Number of results to return
    :param index_path: Path to FAISS index (default: chatboti/data/vectors.faiss)
    :param metadata_path: Path to metadata JSON (default: chatboti/data/metadata.json)
    """
    asyncio.run(search_rag(
        query,
        k,
        index_path if index_path else None,
        metadata_path if metadata_path else None
    ))


@app.command(sort_key=7)
def docker():
    """Build and run Docker container with AWS credentials."""
    run_docker_main()


@app.command(sort_key=8)
def version():
    """Show version."""
    print("chatboti 0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
