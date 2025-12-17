#!/usr/bin/env python3
"""FastAPI server for xConf Assistant's MCP client API."""

import logging
import os
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

import httpx
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from microeval.chat_client import load_config
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from tinyrag.mcp_client import SpeakerMcpClient
from tinyrag.rag import RAGService

model_config = load_config()
chat_models = model_config["chat_models"]
embed_models = model_config["embed_models"]

logger = logging.getLogger(__name__)


class SlimMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    mode: str | None = None
    userToken: str | None = None
    history: list[SlimMessage] | None = None


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.mcp_client: Optional[SpeakerMcpClient] = None
    app.state.ready = False

    chat_service = os.getenv("CHAT_SERVICE")
    embed_service = os.getenv("EMBED_SERVICE") or chat_service

    if not chat_service:
        logger.warning("CHAT_SERVICE not set, skipping MCP initialization")
        app.state.ready = True
    else:
        try:
            # Pre-load embeddings during startup
            logger.info(f"Pre-loading embeddings with {embed_service}...")
            rag_service = RAGService(llm_service=embed_service)
            await rag_service.connect()
            logger.info("Embeddings loaded successfully")

            # Initialize MCP client with chat service
            logger.info(f"Initializing MCP client with {chat_service}...")
            app.state.mcp_client = SpeakerMcpClient(chat_service=chat_service)
            await app.state.mcp_client.connect()
            logger.info("MCP client initialized successfully")
            app.state.ready = True
        except Exception as e:
            logger.error(f"Failed to initialize during startup: {e}")
            raise

    yield

    if app.state.mcp_client:
        try:
            await app.state.mcp_client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting MCP client: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TinyRAG",
        description="Tiny RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        status = {
            "status": "ok",
        }
        if app.state.mcp_client and app.state.mcp_client.tools:
            status["mcp_tools"] = len(app.state.mcp_client.tools)
        return status

    @app.get("/ready")
    async def readiness_check() -> Dict[str, Any]:
        return {"ready": getattr(app.state, "ready", False)}

    @app.get("/info")
    async def get_info() -> Dict[str, Any]:
        info = {}
        if app.state.mcp_client and app.state.mcp_client.chat_client:
            info["chat_service"] = app.state.mcp_client.chat_service
            info["chat_model"] = getattr(
                app.state.mcp_client.chat_client, "model", None
            ) or chat_models.get(app.state.mcp_client.chat_service)
            embed_service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
            info["embed_service"] = embed_service
            info["embed_model"] = embed_models.get(embed_service)
        return info

    @app.get("/")
    async def simple():
        try:
            index_path = Path(__file__).parent / "index.html"
            return FileResponse(str(index_path), media_type="text/html")
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return {"error": "Could not load UI"}

    @app.post("/chat")
    @limiter.limit("10/minute")
    async def chat(request: Request, chat_request: ChatRequest) -> Dict[str, Any]:
        """
        Returns:
            {
                "id": <uuid4>,
                "role": "assistant",
                "status": "success",
                "data": <response from MCP client>
            }
        """
        if not app.state.mcp_client:
            raise HTTPException(status_code=503, detail="Chat service not initialized")

        try:
            history = (
                [msg.model_dump() for msg in chat_request.history]
                if chat_request.history
                else None
            )
            result = await app.state.mcp_client.process_query(
                chat_request.query, history=history
            )
            return {
                "id": str(uuid4()),
                "role": "assistant",
                "status": "success",
                "data": result,
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def wait_and_open_browser(check_url: str, open_url: str):
    max_retries = 60
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = httpx.get(check_url, timeout=1)
            if response.status_code == 200:
                webbrowser.open(open_url)
                print(f"Opening {open_url} in browser...")
                return
        except Exception:
            pass

        time.sleep(0.5)
        retry_count += 1

    try:
        webbrowser.open(open_url)
        print(f"Opening {open_url} in browser (timeout waiting for ready)...")
    except Exception as e:
        print(f"Could not open browser: {e}")


def run_server(host: str, port: int, open_browser: bool = False, reload: bool = False):
    if open_browser:
        base_url = f"http://{host}:{port}"
        thread = threading.Thread(
            target=wait_and_open_browser,
            args=(f"{base_url}/ready", base_url),
            daemon=True,
        )
        thread.start()

    logger.info(f"Starting TinyRAG server on http://{host}:{port}")
    if reload:
        uvicorn.run(
            "tinyrag.fastapi_server:app",
            host=host,
            port=port,
            log_config=None,
            reload=True,
        )
    else:
        fastapi_app = create_app()
        uvicorn.run(fastapi_app, host=host, port=port, log_config=None)


app = create_app()
