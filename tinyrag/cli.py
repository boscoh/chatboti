#!/usr/bin/env python3
"""
TinyRAG CLI - Command-line interface for TinyRAG
"""

import asyncio
import logging
import os
import threading
import time
import webbrowser

import httpx
import typer
import uvicorn

from tinyrag.fastapi_server import create_app
from tinyrag.mcp_client import amain as mcp_amain
from tinyrag.rag import amain as rag_amain
from tinyrag.run_docker import main as run_docker_main
from tinyrag.setup_logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def wait_and_open_browser(check_url: str, open_url: str):
    max_retries = 60
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = httpx.get(check_url, timeout=1)
            if response.status_code == 200:
                webbrowser.open(open_url)
                typer.echo(f"Opening {open_url} in browser...")
                return
        except Exception:
            pass

        time.sleep(0.5)
        retry_count += 1

    try:
        webbrowser.open(open_url)
        typer.echo(f"Opening {open_url} in browser (timeout waiting for ready)...")
    except Exception as e:
        typer.echo(f"Could not open browser: {e}")


def run_server(host: str, port: int, open_browser: bool = False):
    if open_browser:
        base_url = f"http://{host}:{port}"
        thread = threading.Thread(
            target=wait_and_open_browser,
            args=(f"{base_url}/ready", base_url),
            daemon=True,
        )
        thread.start()

    logger.info(f"Starting TinyRAG server on http://{host}:{port}")
    fastapi_app = create_app()
    uvicorn.run(fastapi_app, host=host, port=port, log_config=None)


app = typer.Typer(
    name="tinyrag", help="TinyRAG - Tiny RAG starter kit", no_args_is_help=True
)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start the UI with FastAPI backend and open browser"""
    run_server(host, port, open_browser=True)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start the FastAPI server only"""
    run_server(host, port, open_browser=False)


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
