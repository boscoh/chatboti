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
from microeval.llm import SimpleLLMClient
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chatboti.agent import InfoAgent
from chatboti.config import (
    get_chat_client,
    get_chat_service,
    get_embed_client,
    get_embed_service,
)
from chatboti.faiss_rag import FaissRAGService
from chatboti.utils import get_version

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
    app.state.info_agent: Optional[InfoAgent] = None
    app.state.chat_client: Optional[SimpleLLMClient] = None
    app.state.embed_client: Optional[SimpleLLMClient] = None
    app.state.ready = False
    rag_service: Optional[FaissRAGService] = None

    chat_service = os.getenv("CHAT_SERVICE")

    if not chat_service:
        logger.warning("CHAT_SERVICE not set, skipping MCP initialization")
        app.state.ready = True
        yield
    else:
        try:
            logger.info("Pre-loading embeddings...")

            data_dir = Path(__file__).parent / "data"

            app.state.embed_client = await get_embed_client()

            rag_service = FaissRAGService(
                embed_client=app.state.embed_client, data_dir=data_dir
            )
            await rag_service.__aenter__()
            logger.info(
                f"RAG loaded: {len(rag_service.documents)} documents, {rag_service.index.ntotal} vectors"
            )

            # Manual lifecycle management avoids asyncio task context issues with anyio cancel scopes in Python 3.13
            logger.info("Initializing chat client and MCP agent...")
            app.state.chat_client = await get_chat_client()
            agent = InfoAgent(chat_client=app.state.chat_client)
            await agent.connect()
            app.state.info_agent = agent
            logger.info("MCP client initialized successfully")
            app.state.ready = True

            yield

            logger.info("Shutting down MCP client...")
            if app.state.info_agent:
                await app.state.info_agent.disconnect()
                app.state.info_agent = None

            if rag_service:
                await rag_service.__aexit__(None, None, None)

            if hasattr(app.state, "chat_client") and app.state.chat_client:
                await app.state.chat_client.close()
                app.state.chat_client = None

            if hasattr(app.state, "embed_client") and app.state.embed_client:
                await app.state.embed_client.close()
                app.state.embed_client = None
        except Exception as e:
            logger.error(f"Failed to initialize during startup: {e}")
            raise


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
        if app.state.info_agent and app.state.info_agent.tools:
            status["mcp_tools"] = len(app.state.info_agent.tools)
        return status

    @app.get("/ready")
    async def readiness_check() -> Dict[str, Any]:
        return {"ready": getattr(app.state, "ready", False)}

    @app.get("/info")
    async def get_info() -> Dict[str, Any]:
        info = {"version": get_version()}
        if hasattr(app.state, "chat_client") and app.state.chat_client:
            info["chat_service"] = get_chat_service()
            info["chat_model"] = getattr(app.state.chat_client, "model", "unknown")

        if hasattr(app.state, "embed_client") and app.state.embed_client:
            info["embed_service"] = get_embed_service()
            info["embed_model"] = getattr(app.state.embed_client, "model", "unknown")
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
        """Process chat query via MCP client.

        :param request: FastAPI request
        :param chat_request: Chat query and history
        :return: Dict with id, role, status, and data from MCP client
        """
        if not app.state.info_agent:
            raise HTTPException(status_code=503, detail="Chat service not initialized")

        try:
            history = (
                [msg.model_dump() for msg in chat_request.history]
                if chat_request.history
                else None
            )
            result = await app.state.info_agent.process_query(
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
                logger.info(f"Opening {open_url} in browser...")
                return
        except Exception:
            pass

        time.sleep(0.5)
        retry_count += 1

    try:
        webbrowser.open(open_url)
        logger.info(f"Opening {open_url} in browser (timeout waiting for ready)...")
    except Exception as e:
        logger.error(f"Could not open browser: {e}")


def run_server(host: str, port: int, open_browser: bool = False, reload: bool = False):
    if open_browser:
        base_url = f"http://{host}:{port}"
        thread = threading.Thread(
            target=wait_and_open_browser,
            args=(f"{base_url}/ready", base_url),
            daemon=True,
        )
        thread.start()

    logger.info(f"Starting chatboti {get_version()} on http://{host}:{port}")
    if reload:
        uvicorn.run(
            "chatboti.server:app",
            host=host,
            port=port,
            log_config=None,
            reload=True,
        )
    else:
        fastapi_app = create_app()
        uvicorn.run(fastapi_app, host=host, port=port, log_config=None)


app = create_app()
