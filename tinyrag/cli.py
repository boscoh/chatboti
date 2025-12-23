#!/usr/bin/env python3
"""
TinyRAG CLI - Command-line interface for TinyRAG
"""

import asyncio
import os
import sys

from cyclopts import App

from tinyrag.server import run_server
from tinyrag.agent import amain as agent_amain
from tinyrag.rag import build_embeddings as rag_amain, search_loop
from tinyrag.docker import main as run_docker_main
from tinyrag.logger import setup_logging

setup_logging()


app = App(name="tinyrag", help="TinyRAG - Tiny RAG starter kit")


@app.command(name="ui-chat")
def ui_chat(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the UI with FastAPI backend and open browser.

    :param host: Server host.
    :param port: Server port.
    :param reload: Enable auto-reload on file changes.
    """
    run_server(host, port, open_browser=True, reload=reload)


@app.command
def server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server only.

    :param host: Server host.
    :param port: Server port.
    :param reload: Enable auto-reload on file changes.
    """
    run_server(host, port, open_browser=False, reload=reload)


@app.command(name="cli-chat")
def cli_chat():
    """Interactive chat with agent using MCP tools."""
    service = os.getenv("CHAT_SERVICE")
    if not service:
        print("Error: CHAT_SERVICE environment variable is not set")
        sys.exit(1)
    asyncio.run(agent_amain(service))


@app.command
def rag():
    """Generate embeddings for RAG."""
    asyncio.run(rag_amain())


@app.command
def search():
    """Interactive search loop - query the RAG database directly."""
    asyncio.run(search_loop())


@app.command
def docker():
    """Build and run Docker container with AWS credentials."""
    run_docker_main()


@app.command
def version():
    """Show version."""
    print("tinyrag 0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
