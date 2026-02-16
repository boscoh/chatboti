# Chatboti

A simple chatbot example demonstrating how to build an AI agent that uses MCP (Model Context Protocol) to query a RAG database. Includes CLI, web UI, and Docker deployment options.

## What It Does

An LLM-powered agent answers questions by searching a simple speaker database using semantic search. The agent uses MCP tools to find the best matching speaker for any topic.

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

Supported services: `openai`, `bedrock`

## Three Ways to Run

### 1. Web UI

```bash
uv run chatboti ui-chat
```

Opens a browser with an interactive chat interface.

### 2. CLI Chat

```bash
uv run chatboti cli-chat
```

Interactive terminal chat with the agent.

### 3. Docker

```bash
uv run chatboti docker
```

Builds and runs a Docker container. The `Dockerfile` is configured for CI/ECS deployment with:
- Port `80`
- Health endpoint at `/health`

**AWS Credentials for Bedrock:**
When using `CHAT_SERVICE=bedrock` or `EMBED_SERVICE=bedrock`, CLI automatically:
- Extracts AWS credentials from your `AWS_PROFILE` (or default profile)
- Injects them into Docker container as environment variables
- Validates credentials before building image

No manual credential configuration needed - just ensure your AWS profile is configured locally.

## How It Works

```
User Query → Agent (LLM) → MCP Tools → RAG Database → Response
```

1. **Agent** - An LLM that decides when to use tools
2. **MCP Server** - Provides tools via Model Context Protocol
3. **RAG Service** - Semantic search using embeddings on speaker data

## Project Structure

```
chatboti/
├── cli.py           # CLI commands
├── agent.py         # LLM agent with MCP client
├── mcp_server.py    # MCP server with RAG tools
├── rag.py           # Embeddings and semantic search
├── server.py        # FastAPI web server
├── index.html       # Web UI
└── data/            # Speaker database (CSV + embeddings)
```

## API Endpoints

- `/` - Web chat interface
- `/chat` - Chat API endpoint
- `/health` - Health check
- `/info` - Service configuration

## Environment Variables

| Variable         | Description                                       |
| ---------------- | ------------------------------------------------- |
| `CHAT_SERVICE`   | LLM provider: `openai` or `bedrock` (required)    |
| `EMBED_SERVICE`  | Embedding provider (defaults to `CHAT_SERVICE`)   |
| `OPENAI_API_KEY` | OpenAI API key                                    |
| `AWS_PROFILE`    | AWS profile for Bedrock                           |

## Model Configuration

The project supports multiple LLM and embedding models configured in `summary-evals/models.json`:

**Chat Models:**
- **OpenAI:** GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Bedrock:** Amazon Nova Pro
- **Ollama:** Llama 3.2
- **Groq:** Llama 3.3-70b-versatile

**Embedding Models:**
- **OpenAI:** text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- **Ollama:** nomic-embed-text, mxbai-embed-large, all-minilm
- **Bedrock:** Amazon Titan Embed Text v2 & v1, Cohere Embed v3

## Recent Improvements

### Test Suite Refactoring

- **Reduced Mock Usage:** Replaced mock-based tests with integration tests using real services
- **Consolidated Test Utilities:** Created shared `conftest.py` with deterministic embed client
- **Parameterized Tests:** Added unified search tests for both FAISS and HDF5 backends
- **Improved Service Creation:** Abstracted RAG service creation with unified `create_rag_service()` function

### Storage Backends

The project supports multiple storage backends for flexibility:

- **FAISS + JSON:** Standard multi-file format (`.faiss` + `.json`)
- **HDF5:** Single-file format (`.h5`) with compression and partial loading

Both backends are fully tested and supported in production.
