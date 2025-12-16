# TinyRAG

A lightweight Retrieval Augmented Generation (RAG) starter kit with FastAPI, MCP (Model Context Protocol), and multi-LLM support.

## Features

- **RAG with Semantic Search** - Retrieve relevant information using embeddings
- **Multi-LLM Support** - OpenAI, AWS Bedrock, or local Ollama
- **MCP Integration** - Use Model Context Protocol for tool integration
- **Web UI** - Interactive chat interface
- **Docker Ready** - Containerized deployment

## Quick Start

### Prerequisites

- Python 3.13+
- `uv` package manager

### Installation

```bash
uv sync
```

### Configuration

Create a `.env` file:

```bash
CHAT_SERVICE=openai
OPENAI_API_KEY=your-api-key-here
```

Supported `CHAT_SERVICE` values:
- `openai` - OpenAI API
- `bedrock` - AWS Bedrock

Optional:
- `EMBED_SERVICE` - Service for embeddings (defaults to `CHAT_SERVICE`)
- `LLM_SERVICE` - Legacy variable, deprecated (use `CHAT_SERVICE` instead)

## Usage

### Start the Web UI

```bash
uv run tinyrag ui
```

Options:
- `--host` - Server host (default: 127.0.0.1)
- `--port` - Server port (default: 8000)
- `--no-browser` - Don't open browser automatically

### Start the FastAPI Server

```bash
uv run tinyrag server
```

Options:
- `--host` - Server host (default: 0.0.0.0)
- `--port` - Server port (default: 8000)

### Start the MCP Client

```bash
uv run tinyrag mcp
```

Interactive chat interface using MCP tools. Requires `CHAT_SERVICE` environment variable.

### Build and Run Docker

```bash
uv run tinyrag docker
```

Builds and runs the Docker container with AWS credentials.

### Show Version

```bash
uv run tinyrag version
```

## API Endpoints

### `/` - Web Interface
The main chat interface.

### `/info` - Service Information
Returns LLM service, chat model, and embedding model details:
```json
{
  "llm_service": "openai",
  "chat_model": "gpt-4o",
  "embed_model": "text-embedding-3-small"
}
```

### `/health` - Health Check
Returns server status.

### `/chat` - Chat API
Process a query using the RAG system.

**Request:**
```json
{
  "query": "Find speakers on machine learning",
  "mode": "assistant",
  "history": []
}
```

**Response:**
```json
{
  "id": "uuid",
  "role": "assistant",
  "status": "success",
  "data": "Response text"
}
```

## Configuration

### Supported Models

**OpenAI:**
- Chat: `gpt-4o`
- Embeddings: `text-embedding-3-small`

**AWS Bedrock:**
- Chat: `amazon.nova-pro-v1:0`
- Embeddings: `amazon.titan-embed-text-v2:0`

### Environment Variables

- `CHAT_SERVICE` - Service for chat/MCP operations (`openai` or `bedrock`) (required)
- `EMBED_SERVICE` - Service for embeddings (defaults to `CHAT_SERVICE`)
- `OPENAI_API_KEY` - OpenAI API key (if using OpenAI)
- `AWS_PROFILE` - AWS profile (if using Bedrock)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials (alternative to profile)
- `CORS_OFF` - Disable CORS if set
- `AWS_REGION` - AWS region (default: us-east-1)
- `LLM_SERVICE` - Legacy variable (deprecated, use `CHAT_SERVICE`)

## Project Structure

```
tinyrag/
├── cli.py               # Command-line interface
├── fastapi_server.py    # FastAPI application
├── mcp_server.py        # MCP server implementation
├── mcp_client.py        # MCP client with LLM integration
├── rag.py               # RAG service with embeddings
├── config.py            # Model configuration
├── index.html           # Web UI
└── data/                # Speaker data and embeddings
```

## Architecture

- **FastAPI Server** - REST API and web interface
- **MCP Server** - Tool provider via STDIO
- **MCP Client** - Tool caller with multi-step reasoning
- **RAG Service** - Embeddings and semantic search
- **Chat Client** - Unified LLM interface

## License

MIT
