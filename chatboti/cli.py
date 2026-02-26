#!/usr/bin/env python3
"""Chatboti CLI - Command-line interface for Chatboti"""

import asyncio
import logging
import os
from pathlib import Path

from cyclopts import App

from chatboti.agent import amain as agent_amain
from chatboti.docker import main as run_docker_main
from chatboti.logger import setup_logging
from chatboti.rag_cli import (
    build_embeddings as build_rag_new,
    convert_from_hdf5,
    convert_to_hdf5,
    search_rag,
    show_hdf5_info,
)
from chatboti.config import load_env
from chatboti.server import run_server
from chatboti.utils import get_version

setup_logging()
load_env()

logger = logging.getLogger(__name__)

app = App(name="chatboti", help="Chatboti - RAG starter kit")

_DEFAULT_RAG_MODE = os.getenv("RAG_MODE", "faiss")


@app.command(name="ui-chat", sort_key=0)
def ui_chat(rag_mode: str = _DEFAULT_RAG_MODE):
    """Start the UI with FastAPI backend and open browser.

    :param rag_mode: RAG backend to use (faiss/hdf5/sqlite)
    """
    os.environ["RAG_MODE"] = rag_mode
    run_server("127.0.0.1", 8000, open_browser=True, reload=False)


@app.command(sort_key=1)
def server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    rag_mode: str = _DEFAULT_RAG_MODE,
):
    """Start the FastAPI server only.

    :param host: Server host
    :param port: Server port
    :param reload: Enable auto-reload on file changes
    :param rag_mode: RAG backend to use (faiss/hdf5/sqlite)
    """
    os.environ["RAG_MODE"] = rag_mode
    run_server(host, port, open_browser=False, reload=reload)


@app.command(name="cli-chat", sort_key=2)
def cli_chat(rag_mode: str = _DEFAULT_RAG_MODE):
    """Interactive chat with agent using MCP tools.

    :param rag_mode: RAG backend to use (faiss/hdf5/sqlite)
    """
    os.environ["RAG_MODE"] = rag_mode
    asyncio.run(agent_amain())


@app.command(name="build-rag", sort_key=3)
def build_rag(
    csv_path: str = "",
    index_path: str = "",
    metadata_path: str = "",
    rag_mode: str = _DEFAULT_RAG_MODE,
):
    """Build RAG embeddings from CSV data (defaults to chatboti/data/).

    :param csv_path: Path to CSV file
    :param index_path: Path to save index file (.faiss, .h5, or .db)
    :param metadata_path: Path to save metadata JSON (FAISS only)
    :param rag_mode: RAG backend to use (faiss/hdf5/sqlite)
    """
    asyncio.run(
        build_rag_new(
            csv_path if csv_path else None,
            index_path if index_path else None,
            metadata_path if metadata_path else None,
            rag_mode=rag_mode,
        )
    )


@app.command(name="search-rag", sort_key=4)
def search_rag_cmd(
    query: str,
    k: int = 5,
    index_path: str = "",
    metadata_path: str = "",
    rag_mode: str = _DEFAULT_RAG_MODE,
):
    """Search the RAG index for relevant documents (defaults to chatboti/data/).

    :param query: Search query
    :param k: Number of results to return
    :param index_path: Path to index file (.faiss, .h5, or .db)
    :param metadata_path: Path to metadata JSON (FAISS only)
    :param rag_mode: RAG backend to use (faiss/hdf5/sqlite)
    """
    asyncio.run(
        search_rag(
            query,
            k,
            index_path if index_path else None,
            metadata_path if metadata_path else None,
            rag_mode=rag_mode,
        )
    )


@app.command(name="convert-to-hdf5", sort_key=5)
def convert_to_hdf5_cmd(index_path: str, metadata_path: str, output_path: str):
    """Convert FAISS+JSON format to HDF5 single-file format.

    :param index_path: Path to FAISS index file (.faiss)
    :param metadata_path: Path to metadata JSON file
    :param output_path: Path to output HDF5 file (.h5)
    """
    asyncio.run(convert_to_hdf5(index_path, metadata_path, output_path))


@app.command(name="convert-from-hdf5", sort_key=6)
def convert_from_hdf5_cmd(
    input_path: str, index_path: str = "", metadata_path: str = ""
):
    """Convert HDF5 format to FAISS+JSON format.

    :param input_path: Path to HDF5 file (.h5)
    :param index_path: Path to output FAISS index file
    :param metadata_path: Path to output metadata JSON file
    """
    asyncio.run(
        convert_from_hdf5(
            input_path,
            index_path if index_path else None,
            metadata_path if metadata_path else None,
        )
    )


@app.command(name="hdf5-info", sort_key=7)
def hdf5_info_cmd(hdf5_path: str):
    """Display HDF5 file metadata and statistics.

    :param hdf5_path: Path to HDF5 file (.h5)
    """
    asyncio.run(show_hdf5_info(hdf5_path))


@app.command(sort_key=8)
def docker(rag_mode: str = _DEFAULT_RAG_MODE):
    """Build and run Docker container with AWS credentials.

    :param rag_mode: RAG backend to inject as RAG_MODE into the container (faiss/hdf5/sqlite)
    """
    os.environ["RAG_MODE"] = rag_mode
    run_docker_main()


@app.command(sort_key=9)
def version():
    """Show version."""
    print(f"chatboti {get_version()}")


def main():
    app()


if __name__ == "__main__":
    main()
