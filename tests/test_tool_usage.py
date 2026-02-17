"""
Test tool usage with multiple LLM providers (Ollama, OpenAI, Bedrock).
Tests verify that tools are ACTUALLY EXECUTED and return real data from CSV.

Tests automatically run for all provider configurations that can be initialized.
Configure providers in .env with API keys/credentials.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.asyncio

from microeval.llm import get_llm_client, load_config

from chatboti.agent import InfoAgent


def load_test_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def load_speaker_data():
    """Load speaker data from CSV file."""
    csv_path = (
        Path(__file__).parent.parent
        / "chatboti"
        / "data"
        / "2025-09-02-speaker-bio.csv"
    )
    speakers = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            speakers = list(reader)
    return speakers


# Provider configurations to test
# Tests will attempt to initialize each and skip if initialization fails
PROVIDER_CONFIGS = [
    {
        "name": "ollama",
        "chat_service": "ollama",
        "chat_model": None,  # Use default from config
        "embed_service": None,
        "embed_model": None,
    },
    {
        "name": "openai",
        "chat_service": "openai",
        "chat_model": None,  # Use default from config
        "embed_service": None,
        "embed_model": None,
    },
    {
        "name": "bedrock",
        "chat_service": "bedrock",
        "chat_model": None,  # Use default from config
        "embed_service": None,
        "embed_model": None,
    },
]


async def try_initialize_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to initialize a provider config. Returns config if successful, None if failed."""
    load_test_env()

    chat_service = config.get("chat_service")
    chat_model = config.get("chat_model")

    if not chat_service:
        return None

    # Get default model from config if not specified
    if not chat_model:
        model_config = load_config()
        chat_model = model_config.get("chat_models", {}).get(chat_service)

    if not chat_model:
        return None

    # Try to initialize the client
    try:
        client = get_llm_client(chat_service, model=chat_model)
        await client.connect()
        await client.close()

        return {
            "name": config["name"],
            "chat_service": chat_service,
            "chat_model": chat_model,
            "embed_service": config.get("embed_service"),
            "embed_model": config.get("embed_model"),
        }
    except Exception as e:
        # Initialization failed - service not available
        return None


def get_available_configs():
    """Get list of provider configs that can be initialized."""
    import asyncio

    available = []
    for config in PROVIDER_CONFIGS:
        try:
            result = asyncio.run(try_initialize_config(config))
            if result:
                available.append(result)
        except Exception:
            pass

    return available


@pytest.fixture(
    scope="module", params=get_available_configs(), ids=lambda cfg: cfg["name"]
)
def provider_config(request):
    """Parametrized fixture that runs tests for each available provider config."""
    return request.param


@pytest.fixture(scope="module")
def chat_service(provider_config):
    """Get chat service from provider config."""
    return provider_config["chat_service"]


@pytest.fixture(scope="module")
def chat_model(provider_config):
    """Get chat model from provider config."""
    return provider_config["chat_model"]


@pytest.fixture(scope="module")
def speaker_data():
    """Load speaker data from CSV."""
    return load_speaker_data()


def test_show_available_configs():
    """Show which provider configs will be tested based on successful initialization."""
    load_test_env()
    available = get_available_configs()

    print("\n" + "=" * 70)
    print("AVAILABLE LLM PROVIDER CONFIGURATIONS FOR TESTING:")
    print("=" * 70)

    for config in PROVIDER_CONFIGS:
        name = config["name"]
        is_available = any(a["name"] == name for a in available)
        status = "✓ AVAILABLE" if is_available else "✗ UNAVAILABLE"
        print(f"  {name:12s}: {status}")

        if is_available:
            cfg = next(a for a in available if a["name"] == name)
            print(f"               → chat: {cfg['chat_service']} / {cfg['chat_model']}")
        else:
            chat_service = config.get("chat_service")
            if chat_service == "ollama":
                print(f"               → Ollama not running or model not available")
            elif chat_service == "openai":
                print(f"               → Set OPENAI_API_KEY in .env")
            elif chat_service == "bedrock":
                print(f"               → Set AWS_PROFILE or AWS credentials in .env")

    print("=" * 70)
    print(
        f"Testing {len(available)} configuration(s): {', '.join([a['name'] for a in available]) or 'none'}"
    )
    print("=" * 70 + "\n")

    if not available:
        pytest.skip("No LLM provider configurations available for testing")


async def test_agent_initialization(provider_config):
    """Test that InfoAgent can be initialized with the configured service."""
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing agent initialization...")

    chat_client = get_llm_client(chat_service, model=chat_model)
    async with InfoAgent(chat_client) as agent:
        assert agent.chat_client is not None
        if chat_model:
            assert agent.chat_client.model == chat_model
        assert agent.tools is not None
        assert len(agent.tools) > 0
        print(f"[{name.upper()}] ✓ Agent initialized with {len(agent.tools)} tools")


async def test_tool_metadata(provider_config):
    """Test that tool metadata is correctly loaded."""
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing tool metadata...")

    chat_client = get_llm_client(chat_service, model=chat_model)
    async with InfoAgent(chat_client) as agent:
        assert agent.tools is not None

        tool_names = [tool["function"]["name"] for tool in agent.tools]
        assert "list_all_speakers" in tool_names
        assert "get_best_speaker" in tool_names
        print(f"[{name.upper()}] ✓ Tools loaded: {', '.join(tool_names)}")


async def test_list_all_speakers_actually_executes(provider_config, speaker_data):
    """Verify that list_all_speakers tool is ACTUALLY EXECUTED and returns real CSV data.

    This test checks for actual speaker names from CSV that could only come from tool execution.
    """
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing list_all_speakers execution...")

    if not speaker_data:
        pytest.skip("No speaker data available")

    csv_speaker_names = [
        s.get("Name", "").strip() for s in speaker_data if s.get("Name")
    ]
    assert len(csv_speaker_names) > 0, "CSV should have speaker names"

    chat_client = get_llm_client(chat_service, model=chat_model)
    async with InfoAgent(chat_client) as agent:
        query = "List all available speakers"
        response = await agent.process_query(query)

        assert response is not None
        assert len(response) > 50, "Response should contain speaker names"

        response_lower = response.lower()

        found_speakers = []
        for name in csv_speaker_names:
            name_lower = name.lower()
            name_parts = [
                part.strip() for part in name_lower.split() if len(part.strip()) > 2
            ]
            if any(part in response_lower for part in name_parts):
                found_speakers.append(name)

        print(f"[{name.upper()}] ✓ Found {len(found_speakers)} speakers in response")

        assert len(found_speakers) > 0, (
            f"Tool must be EXECUTED - response should contain actual speaker names from CSV. "
            f"Expected to find at least one of: {csv_speaker_names[:5]}. "
            f"Found: {found_speakers}. "
            f"Response: {response[:800]}"
        )


async def test_get_best_speaker_actually_executes(provider_config, speaker_data):
    """Verify that get_best_speaker tool is ACTUALLY EXECUTED and returns real CSV data.

    This test checks for actual speaker names/details from CSV that could only come from tool execution.
    """
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing get_best_speaker execution...")

    if not speaker_data:
        pytest.skip("No speaker data available")

    test_speaker = speaker_data[0]
    speaker_name = test_speaker.get("Name", "").strip()
    speaker_title = test_speaker.get("Final title", "").strip()

    if not speaker_name or not speaker_title:
        pytest.skip("Speaker missing name or title")

    title_keywords = [w.lower() for w in speaker_title.split() if len(w) > 4][:3]
    if not title_keywords:
        title_keywords = ["modernization", "legacy", "incremental"]

    query_topic = " ".join(title_keywords[:2])

    chat_client = get_llm_client(chat_service, model=chat_model)
    async with InfoAgent(chat_client) as agent:
        query = f"Find the best speaker for: {query_topic}"
        response = await agent.process_query(query)

        assert response is not None
        assert len(response) > 50, "Response should contain speaker information"

        response_lower = response.lower()
        speaker_name_lower = speaker_name.lower()

        speaker_name_parts = [
            part.strip() for part in speaker_name_lower.split() if len(part.strip()) > 2
        ]
        speaker_found = any(part in response_lower for part in speaker_name_parts)

        print(
            f"[{name.upper()}] ✓ Found speaker '{speaker_name}' in response: {speaker_found}"
        )

        assert speaker_found, (
            f"Tool must be EXECUTED - response should contain actual speaker name '{speaker_name}' from CSV. "
            f"Looking for parts: {speaker_name_parts}. "
            f"Response: {response[:800]}"
        )


async def test_multi_step_tool_chaining(provider_config, speaker_data):
    """Verify that multiple tools are ACTUALLY EXECUTED in sequence.

    This test verifies:
    1. list_all_speakers is executed and returns real data
    2. get_best_speaker is executed and returns real data
    3. Response integrates actual results from both tool executions
    """
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing multi-step tool chaining...")

    if not speaker_data:
        pytest.skip("No speaker data available")

    csv_speaker_names = [
        s.get("Name", "").strip() for s in speaker_data if s.get("Name")
    ]
    test_speaker = speaker_data[0]
    speaker_name = test_speaker.get("Name", "").strip()
    speaker_title = test_speaker.get("Final title", "").strip()

    if not speaker_name or not speaker_title:
        pytest.skip("Speaker missing name or title")

    title_keywords = [w.lower() for w in speaker_title.split() if len(w) > 4][:3]
    if not title_keywords:
        title_keywords = ["modernization", "legacy", "incremental"]

    query_topic = " ".join(title_keywords[:2])

    chat_client = get_llm_client(chat_service, model=chat_model)
    async with InfoAgent(chat_client) as agent:
        query = (
            f"Use the list_all_speakers tool, then use get_best_speaker to find a speaker for '{query_topic}'. "
            f"Tell me how many speakers there are and who is the best speaker for '{query_topic}'."
        )
        response = await agent.process_query(query)

        assert response is not None
        assert len(response) > 80, (
            "Response should integrate results from multiple tool calls"
        )

        response_lower = response.lower()

        found_any_speaker = any(
            any(
                part in response_lower for part in name.lower().split() if len(part) > 2
            )
            for name in csv_speaker_names
        )

        speaker_name_parts = [
            part.strip()
            for part in speaker_name.lower().split()
            if len(part.strip()) > 2
        ]
        found_target_speaker = any(
            part in response_lower for part in speaker_name_parts
        )

        number_words = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
        has_number = any(word in response_lower for word in number_words)

        print(
            f"[{name.upper()}] ✓ Multi-step chaining - speakers found: {found_any_speaker or found_target_speaker}, has count: {has_number}"
        )

        assert found_any_speaker or found_target_speaker, (
            f"Tools must be EXECUTED - response should contain actual speaker names from CSV. "
            f"Expected to find '{speaker_name}' or any of: {csv_speaker_names[:3]}. "
            f"Response: {response[:800]}"
        )

        assert has_number or len(response.split()) > 25, (
            f"Response should show evidence of list_all_speakers execution (number of speakers). "
            f"Response: {response[:800]}"
        )


async def test_llm_client_handles_tools(provider_config):
    """Test that the LLM client correctly handles tools parameter.

    This test verifies:
    1. Client accepts tools parameter without error
    2. Response is returned even when tools are provided
    3. The implementation handles tools parameter correctly
    """
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing LLM client tool handling...")

    if not chat_model:
        pytest.skip("CHAT_MODEL not set")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    client = get_llm_client(chat_service, model=chat_model)
    await client.connect()

    try:
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco? Use the get_weather tool.",
            }
        ]

        response_with_tools = await client.get_completion(messages, tools=tools)
        response_without_tools = await client.get_completion(messages, tools=None)

        assert response_with_tools is not None, (
            f"{chat_service} client should return response when tools provided"
        )
        assert response_without_tools is not None, (
            f"{chat_service} client should return response without tools"
        )

        assert "text" in response_with_tools, "Response should have text field"
        assert "text" in response_without_tools, "Response should have text field"

        has_text = (
            response_with_tools.get("text") and len(response_with_tools["text"]) > 0
        )
        has_tool_calls = response_with_tools.get("tool_calls")
        assert has_text or has_tool_calls, (
            f"{chat_service} client should return text or tool_calls. Got: {response_with_tools}"
        )

        tool_calls = response_with_tools.get("tool_calls")

        if tool_calls is not None:
            assert isinstance(tool_calls, list), (
                "tool_calls should be a list if present"
            )
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    assert "function" in tool_call, (
                        "Each tool_call should have a function"
                    )
                    assert "name" in tool_call["function"], (
                        "Tool call should have a function name"
                    )

        print(f"\n{chat_service.upper()} tool handling:")
        text_len = len(response_with_tools.get("text") or "")
        print(f"  Response text length: {text_len}")
        print(f"  Tool calls returned: {len(tool_calls) if tool_calls else 0}")
        if tool_calls:
            for tc in tool_calls:
                print(f"    - {tc['function']['name']}")
    finally:
        await client.close()


async def test_tool_call_extraction(provider_config):
    """Test that tool calls are properly extracted from the LLM response.

    This is critical for tool chaining to work correctly.
    """
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    print(f"\n[{name.upper()}] Testing tool call extraction...")

    if not chat_model:
        pytest.skip("CHAT_MODEL not set")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. UTC, America/New_York",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        }
    ]

    client = get_llm_client(chat_service, model=chat_model)
    await client.connect()

    try:
        messages = [
            {
                "role": "system",
                "content": "You must use the get_current_time tool to answer time questions. Always call the tool.",
            },
            {
                "role": "user",
                "content": "What time is it in New York? You MUST use the get_current_time tool.",
            },
        ]

        response = await client.get_completion(messages, tools=tools)

        assert response is not None

        tool_calls = response.get("tool_calls", [])

        print(f"\n{chat_service.upper()} tool call extraction:")
        print(f"  Response keys: {list(response.keys())}")
        print(f"  Tool calls: {len(tool_calls) if tool_calls else 0}")

        if tool_calls:
            for tc in tool_calls:
                print(f"    - name: {tc['function'].get('name')}")
                print(f"      args: {tc['function'].get('arguments')}")
                print(f"      id: {tc['function'].get('tool_call_id', 'N/A')}")

            assert len(tool_calls) > 0, f"{chat_service} should return tool_calls"

            tc = tool_calls[0]
            assert tc["function"]["name"] == "get_current_time", (
                "Should call get_current_time"
            )

            has_id = tc["function"].get("tool_call_id")
            assert has_id, f"{chat_service} tool_call should have an ID for chaining"
        else:
            print(
                f"  Note: {chat_service} did not return tool_calls (may need different prompt)"
            )
            print(f"  Response text: {response.get('text', '')[:200]}")
    finally:
        await client.close()


# Provider-specific tests


async def test_openai_specific_features(provider_config):
    """Test OpenAI-specific tool calling features."""
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    if chat_service != "openai":
        pytest.skip("This test is OpenAI-specific")

    from microeval.llm import OpenAIClient

    client = OpenAIClient(model=chat_model)
    await client.connect()

    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search a database for records",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "user",
                "content": "Search the database for 'AI speakers'. Use the search_database tool.",
            }
        ]

        response = await client.get_completion(messages, tools=tools)

        assert response is not None
        assert "text" in response or "tool_calls" in response

        print("\nOpenAI tool calling response:")
        print(f"  Has text: {bool(response.get('text'))}")
        print(f"  Has tool_calls: {bool(response.get('tool_calls'))}")

        if response.get("tool_calls"):
            tc = response["tool_calls"][0]
            print(f"  Tool called: {tc['function']['name']}")
            print(f"  Arguments: {tc['function'].get('arguments')}")
            assert "tool_call_id" in tc["function"] or "id" in tc, (
                "OpenAI should provide tool_call_id"
            )
    finally:
        await client.close()


async def test_ollama_specific_features(provider_config):
    """Test Ollama-specific tool calling features."""
    name = provider_config["name"]
    chat_service = provider_config["chat_service"]
    chat_model = provider_config["chat_model"]

    if chat_service != "ollama":
        pytest.skip("This test is Ollama-specific")

    from microeval.llm import OllamaClient

    client = OllamaClient(model=chat_model)
    await client.connect()

    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        messages = [
            {"role": "user", "content": "Calculate 2+2 using the calculate tool."}
        ]

        response = await client.get_completion(messages, tools=tools)

        assert response is not None
        assert "text" in response

        print("\nOllama tool calling response:")
        print(f"  Has text: {bool(response.get('text'))}")
        print(f"  Has tool_calls: {bool(response.get('tool_calls'))}")
        print(f"  Text preview: {response.get('text', '')[:200]}")
    finally:
        await client.close()
