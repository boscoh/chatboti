# Microeval Environment Variable Loading - Complete Trace

## Overview
Microeval uses python-dotenv to load environment variables from `.env` files.
The loading happens in `microeval/cli.py` at module import time.

## Loading Flow

### 1. **Module Import Time** (microeval/cli.py:25)
```python
from dotenv import load_dotenv
load_dotenv()  # Called at module import, no parameters
```

### 2. **dotenv Default Behavior**
When `load_dotenv()` is called with no parameters:
- Searches for `.env` file starting from **current working directory (CWD)**
- Walks UP the directory tree until it finds `.env` or reaches root
- **Does NOT override** existing environment variables (override=False by default)

### 3. **Search Path Priority**
```
Current Working Directory (.env)
    ↓ (if not found, go up)
Parent Directory (../.env)
    ↓ (if not found, go up)
Grandparent Directory (../../.env)
    ↓ (continue until found or reach root)
```

### 4. **Variable Priority** (Highest to Lowest)
1. **System environment variables** (export FOO=bar, already in shell)
2. **Command-line exports** (FOO=bar microeval ...)
3. **.env file in CWD** (first found going upward)
4. **Default values in code** (hardcoded config)

## UVX Behavior

### When running `uvx microeval ...`:
1. **uvx creates an isolated virtual environment** in `~/.local/share/uv/tools/microeval/`
2. **CWD remains unchanged** - stays at your current directory
3. **microeval executes from CWD**, so `.env` search starts there
4. **System env vars are inherited** from your shell

### Example Flow:
```bash
$ pwd
/Users/boscoh/p/chatboti

$ cat .env
CHAT_SERVICE=openai
EMBED_SERVICE=ollama
OPENAI_API_KEY=sk-...

$ uvx microeval chat "hello"
# Flow:
# 1. uvx installs/uses microeval in isolated env
# 2. microeval runs from /Users/boscoh/p/chatboti (CWD)
# 3. load_dotenv() finds .env in CWD
# 4. Loads CHAT_SERVICE, EMBED_SERVICE, OPENAI_API_KEY
# 5. Inherits any system env vars (AWS_PROFILE, etc.)
```

## Code References

### microeval/cli.py
```python
from dotenv import load_dotenv

load_dotenv()  # Line 25 - loads from CWD/.env
```

### microeval/llm.py (Client initialization)
```python
# OpenAI client (line ~594)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in .env or environment")

# Groq client (line ~776)
api_key = os.getenv("GROQ_API_KEY")

# Bedrock client (line ~892)
profile_name = os.getenv("AWS_PROFILE")
region = os.getenv("AWS_REGION")
```

## Key Behaviors

### ✓ System env vars take precedence
```bash
export CHAT_SERVICE=bedrock  # System var
# .env has CHAT_SERVICE=openai
uvx microeval chat "hi"  # Uses bedrock (system wins)
```

### ✓ .env file location matters
```bash
/project/.env         # Will be found
/project/subdir/      # CWD - load_dotenv() walks UP to /project/.env
/home/user/.env       # Won't be found unless CWD is /home/user
```

### ✓ uvx doesn't change CWD
```bash
cd /my/project
uvx microeval ...  # Still runs from /my/project, finds /my/project/.env
```

### ✓ Comments and spacing in .env files
```bash
# Comments are supported with #
CHAT_SERVICE=openai # inline comments work
EMBED_SERVICE=ollama
# Spaces around = are trimmed
KEY = value  # equivalent to KEY=value
```

## Environment Variables Used by Chatboti

From `.env` file:
- `CHAT_SERVICE` - Which LLM service to use (openai, ollama, bedrock, groq)
- `CHAT_MODEL` - Optional, overrides default model
- `EMBED_SERVICE` - Which embedding service to use
- `EMBED_MODEL` - Optional, embedding model name
- `OPENAI_API_KEY` - Required for OpenAI
- `GROQ_API_KEY` - Required for Groq
- `AWS_PROFILE` - Required for Bedrock
- `AWS_REGION` - Required for Bedrock (e.g., us-east-1)

## Debugging

### Check what .env file is loaded:
```python
from dotenv import find_dotenv
import os
print(f"CWD: {os.getcwd()}")
print(f".env found at: {find_dotenv(usecwd=True)}")
```

### Verify environment variables are loaded:
```python
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)  # Shows which .env file was loaded
print(f"CHAT_SERVICE: {os.getenv('CHAT_SERVICE')}")
print(f"EMBED_SERVICE: {os.getenv('EMBED_SERVICE')}")
```

### Check environment variables from shell:
```bash
env | grep -E "CHAT_SERVICE|EMBED_SERVICE|OPENAI_API_KEY"
```

### Override for testing:
```bash
CHAT_SERVICE=ollama uvx microeval chat "hello"  # Temporary override
```

## Common Issues

### .env file not found
**Problem**: Variables not loading
**Solution**: Check you're running from correct directory
```bash
pwd  # Should be project root with .env
ls -la .env  # Should exist
```

### System env vars taking precedence
**Problem**: .env changes not taking effect
**Solution**: Unset system env vars or use override=True
```python
load_dotenv(override=True)  # Force .env to override system
```

### Different .env for different environments
**Solution**: Use different env files
```bash
load_dotenv('.env.development')  # Development
load_dotenv('.env.production')   # Production
```

## How Chatboti Uses This

Chatboti services (server, MCP, CLI) all call `load_dotenv()` early:

```python
# chatboti/server.py, chatboti/mcp_server.py, chatboti/rag_cli.py
from dotenv import load_dotenv

load_dotenv()  # Loads from CWD/.env

# Then later:
from microeval.llm import get_llm_client

client = get_llm_client(
    os.getenv("EMBED_SERVICE"),  # From .env or system
    model=os.getenv("EMBED_MODEL")  # From .env or system
)
```

This ensures consistent environment loading across all entry points.
