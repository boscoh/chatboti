#!/usr/bin/env python3
"""
TinyRAG CLI - Command-line interface for TinyRAG
"""

import asyncio
import os

import typer

from tinyrag.fastapi_server import run_server
from tinyrag.mcp_client import amain as mcp_amain
from tinyrag.rag import amain as rag_amain
from tinyrag.run_docker import main as run_docker_main
from tinyrag.setup_logger import setup_logging

setup_logging()


app = typer.Typer(
    name="tinyrag", help="TinyRAG - Tiny RAG starter kit", no_args_is_help=True
)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload on file changes"),
):
    """Start the UI with FastAPI backend and open browser"""
    run_server(host, port, open_browser=True, reload=reload)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload on file changes"),
):
    """Start the FastAPI server only"""
    run_server(host, port, open_browser=False, reload=reload)


@app.command()
def mcp():
    """Start MCP client"""
    service = os.getenv("CHAT_SERVICE")
    if not service:
        typer.echo("Error: CHAT_SERVICE environment variable is not set")
        raise typer.Exit(1)
    asyncio.run(mcp_amain(service))


@app.command()
def rag():
    """Generate embeddings for RAG"""
    asyncio.run(rag_amain())


@app.command()
def docker():
    """Build and run Docker container with AWS credentials"""
    run_docker_main()


@app.command()
def version():
    """Show version"""
    typer.echo("tinyrag 0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
