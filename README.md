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
When using `CHAT_SERVICE=bedrock` or `EMBED_SERVICE=bedrock`, the CLI automatically:
- Extracts AWS credentials from your `AWS_PROFILE` (or default profile)
- Injects them into the Docker container as environment variables
- Validates credentials before building the image

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
