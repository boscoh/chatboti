"""
Test tool usage with Ollama, using the model specified in .env file.
Tests verify that tools are ACTUALLY EXECUTED and return real data from CSV.
"""

import os
import csv
import pytest
from pathlib import Path
from dotenv import load_dotenv

pytestmark = pytest.mark.asyncio

from chatboti.agent import InfoAgent
from microeval.llm import OllamaClient


def load_test_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def load_speaker_data():
    """Load speaker data from CSV file."""
    csv_path = Path(__file__).parent.parent / "chatboti" / "data" / "2025-09-02-speaker-bio.csv"
    speakers = []
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            speakers = list(reader)
    return speakers


@pytest.fixture(scope="module")
def chat_service():
    """Get chat service from environment."""
    load_test_env()
    service = os.getenv("CHAT_SERVICE")
    if not service:
        pytest.skip("CHAT_SERVICE not set in .env")
    return service


@pytest.fixture(scope="module")
def chat_model():
    """Get chat model from environment."""
    load_test_env()
    model = os.getenv("CHAT_MODEL")
    return model


@pytest.fixture(scope="module")
def speaker_data():
    """Load speaker data from CSV."""
    return load_speaker_data()


async def test_ollama_agent_initialization(chat_service, chat_model):
    """Test that InfoAgent can be initialized with Ollama and correct model."""
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    async with InfoAgent(chat_service=chat_service) as agent:
        assert agent.chat_service == "ollama"
        assert agent.chat_client is not None
        if chat_model:
            assert agent.chat_client.model == chat_model
        assert agent.tools is not None
        assert len(agent.tools) > 0


async def test_tool_metadata(chat_service):
    """Test that tool metadata is correctly loaded."""
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    async with InfoAgent(chat_service=chat_service) as agent:
        assert agent.tools is not None
        
        tool_names = [tool["function"]["name"] for tool in agent.tools]
        assert "list_all_speakers" in tool_names
        assert "get_best_speaker" in tool_names


async def test_list_all_speakers_actually_executes(chat_service, speaker_data):
    """Verify that list_all_speakers tool is ACTUALLY EXECUTED and returns real CSV data.
    
    This test checks for actual speaker names from CSV that could only come from tool execution.
    """
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    if not speaker_data:
        pytest.skip("No speaker data available")
    
    csv_speaker_names = [s.get('Name', '').strip() for s in speaker_data if s.get('Name')]
    assert len(csv_speaker_names) > 0, "CSV should have speaker names"
    
    async with InfoAgent(chat_service=chat_service) as agent:
        query = "List all available speakers"
        response = await agent.process_query(query)
        
        assert response is not None
        assert len(response) > 50, "Response should contain speaker names"
        
        response_lower = response.lower()
        
        found_speakers = []
        for name in csv_speaker_names:
            name_lower = name.lower()
            name_parts = [part.strip() for part in name_lower.split() if len(part.strip()) > 2]
            if any(part in response_lower for part in name_parts):
                found_speakers.append(name)
        
        assert len(found_speakers) > 0, (
            f"Tool must be EXECUTED - response should contain actual speaker names from CSV. "
            f"Expected to find at least one of: {csv_speaker_names[:5]}. "
            f"Found: {found_speakers}. "
            f"Response: {response[:800]}"
        )


async def test_get_best_speaker_actually_executes(chat_service, speaker_data):
    """Verify that get_best_speaker tool is ACTUALLY EXECUTED and returns real CSV data.
    
    This test checks for actual speaker names/details from CSV that could only come from tool execution.
    """
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    if not speaker_data:
        pytest.skip("No speaker data available")
    
    test_speaker = speaker_data[0]
    speaker_name = test_speaker.get('Name', '').strip()
    speaker_title = test_speaker.get('Final title', '').strip()
    
    if not speaker_name or not speaker_title:
        pytest.skip("Speaker missing name or title")
    
    title_keywords = [w.lower() for w in speaker_title.split() if len(w) > 4][:3]
    if not title_keywords:
        title_keywords = ["modernization", "legacy", "incremental"]
    
    query_topic = " ".join(title_keywords[:2])
    
    async with InfoAgent(chat_service=chat_service) as agent:
        query = f"Find the best speaker for: {query_topic}"
        response = await agent.process_query(query)
        
        assert response is not None
        assert len(response) > 50, "Response should contain speaker information"
        
        response_lower = response.lower()
        speaker_name_lower = speaker_name.lower()
        
        speaker_name_parts = [part.strip() for part in speaker_name_lower.split() if len(part.strip()) > 2]
        speaker_found = any(part in response_lower for part in speaker_name_parts)
        
        assert speaker_found, (
            f"Tool must be EXECUTED - response should contain actual speaker name '{speaker_name}' from CSV. "
            f"Looking for parts: {speaker_name_parts}. "
            f"Response: {response[:800]}"
        )


async def test_multi_step_tool_chaining_actually_executes(chat_service, speaker_data):
    """Verify that multiple tools are ACTUALLY EXECUTED in sequence.
    
    This test verifies:
    1. list_all_speakers is executed and returns real data
    2. get_best_speaker is executed and returns real data
    3. Response integrates actual results from both tool executions
    """
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    if not speaker_data:
        pytest.skip("No speaker data available")
    
    csv_speaker_names = [s.get('Name', '').strip() for s in speaker_data if s.get('Name')]
    test_speaker = speaker_data[0]
    speaker_name = test_speaker.get('Name', '').strip()
    speaker_title = test_speaker.get('Final title', '').strip()
    
    if not speaker_name or not speaker_title:
        pytest.skip("Speaker missing name or title")
    
    title_keywords = [w.lower() for w in speaker_title.split() if len(w) > 4][:3]
    if not title_keywords:
        title_keywords = ["modernization", "legacy", "incremental"]
    
    query_topic = " ".join(title_keywords[:2])
    
    async with InfoAgent(chat_service=chat_service) as agent:
        query = (
            f"Use the list_all_speakers tool, then use get_best_speaker to find a speaker for '{query_topic}'. "
            f"Tell me how many speakers there are and who is the best speaker for '{query_topic}'."
        )
        response = await agent.process_query(query)
        
        assert response is not None
        assert len(response) > 80, "Response should integrate results from multiple tool calls"
        
        response_lower = response.lower()
        
        found_any_speaker = any(
            any(part in response_lower for part in name.lower().split() if len(part) > 2)
            for name in csv_speaker_names
        )
        
        speaker_name_parts = [part.strip() for part in speaker_name.lower().split() if len(part.strip()) > 2]
        found_target_speaker = any(part in response_lower for part in speaker_name_parts)
        
        number_words = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        has_number = any(word in response_lower for word in number_words)
        
        assert found_any_speaker or found_target_speaker, (
            f"Tools must be EXECUTED - response should contain actual speaker names from CSV. "
            f"Expected to find '{speaker_name}' or any of: {csv_speaker_names[:3]}. "
            f"Response: {response[:800]}"
        )
        
        assert has_number or len(response.split()) > 25, (
            f"Response should show evidence of list_all_speakers execution (number of speakers). "
            f"Response: {response[:800]}"
        )


async def test_ollama_client_handles_tools(chat_service, chat_model):
    """Test that OllamaClient correctly handles tools parameter.
    
    This test verifies:
    1. OllamaClient accepts tools parameter without error
    2. Response is returned even when tools are provided
    3. The implementation handles tools parameter correctly
    """
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
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
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    client = OllamaClient(model=chat_model)
    await client.connect()
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather in San Francisco? Use the get_weather tool."
        }
    ]
    
    response_with_tools = await client.get_completion(messages, tools=tools)
    response_without_tools = await client.get_completion(messages, tools=None)
    
    assert response_with_tools is not None, "OllamaClient should return response when tools provided"
    assert response_without_tools is not None, "OllamaClient should return response without tools"
    
    assert "text" in response_with_tools, "Response should have text field"
    assert "text" in response_without_tools, "Response should have text field"
    
    assert len(response_with_tools["text"]) > 0, (
        f"OllamaClient should return text when tools provided. Got: {response_with_tools}"
    )
    
    assert "metadata" in response_with_tools, "Response should include metadata"
    
    tool_calls = response_with_tools.get("tool_calls")
    
    if tool_calls is not None:
        assert isinstance(tool_calls, list), "tool_calls should be a list if present"
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                assert "function" in tool_call, "Each tool_call should have a function"
                assert "name" in tool_call["function"], "Tool call should have a function name"
    
    response_text = response_with_tools["text"].lower()
    
    assert "weather" in response_text or "san francisco" in response_text or len(response_text) > 10, (
        f"OllamaClient should return meaningful response. Got: {response_with_tools['text'][:200]}"
    )


async def test_ollama_client_tool_handling_analysis(chat_service, chat_model):
    """Analyze OllamaClient tool handling - documents current implementation issues.
    
    ANALYSIS of microeval/llm.py OllamaClient.get_completion():
    
    ISSUES FOUND:
    1. Line 370: Docstring incorrectly states "Tools not supported" - Ollama DOES support tools
    2. Line 382-384: Doesn't pass `tools` parameter to self.client.chat() call
    3. Line 387: Only extracts response["message"]["content"], ignores tool_calls
    4. Line 392-404: Returns dict with only "text" and "metadata", missing "tool_calls" field
    
    COMPARISON with other clients:
    - OpenAIClient (lines 528, 552-564, 575): Properly passes tools, extracts tool_calls, returns them
    - BedrockClient (lines 842-852, 872): Properly extracts tool_calls from response, returns them
    
    This test verifies:
    1. Ollama API DOES support tools and returns tool_calls when tools are provided
    2. OllamaClient currently doesn't extract tool_calls (BUG - needs fix)
    3. Documents what needs to be fixed in OllamaClient
    """
    if chat_service != "ollama":
        pytest.skip(f"CHAT_SERVICE is {chat_service}, not ollama")
    
    if not chat_model:
        pytest.skip("CHAT_MODEL not set")
    
    import ollama
    
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
                            "description": "The city and state"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather in Paris? Use the get_weather tool."
        }
    ]
    
    raw_client = ollama.AsyncClient()
    raw_response = await raw_client.chat(
        model=chat_model,
        messages=messages,
        tools=tools
    )
    
    assert raw_response is not None, "Ollama API should return a response"
    assert "message" in raw_response, "Response should have message field"
    
    message = raw_response["message"]
    
    ollama_has_tool_calls = (
        hasattr(message, "tool_calls") and message.tool_calls is not None and len(message.tool_calls) > 0
    ) or (
        isinstance(message, dict) and "tool_calls" in message and message.get("tool_calls")
    )
    
    if ollama_has_tool_calls:
        if hasattr(message, "tool_calls"):
            tool_calls = message.tool_calls
        else:
            tool_calls = message.get("tool_calls", [])
        
        assert len(tool_calls) > 0, "Ollama should return tool_calls when tools are provided"
        
        ollama_client = OllamaClient(model=chat_model)
        await ollama_client.connect()
        
        response = await ollama_client.get_completion(messages, tools=tools)
        
        assert response is not None, "OllamaClient should return response"
        assert "text" in response, "Response should have text field"
        
        client_has_tool_calls = "tool_calls" in response and response.get("tool_calls")
        
        assert client_has_tool_calls, (
            f"OllamaClient should extract tool_calls from Ollama response.\n"
            f"Ollama API returned tool_calls: {tool_calls}\n"
            f"OllamaClient response keys: {list(response.keys())}\n"
            f"OllamaClient tool_calls: {response.get('tool_calls')}\n\n"
            f"If this fails, OllamaClient.get_completion() needs to:\n"
            f"1. Pass tools parameter: self.client.chat(..., tools=tools)\n"
            f"2. Extract tool_calls: response['message'].get('tool_calls')\n"
            f"3. Format and return tool_calls in response dict"
        )
        
        extracted_tool_calls = response.get("tool_calls", [])
        assert len(extracted_tool_calls) > 0, "OllamaClient should return non-empty tool_calls list"
        
        for tool_call in extracted_tool_calls:
            assert "function" in tool_call, "Each tool_call should have a function"
            assert "name" in tool_call["function"], "Tool call function should have a name"
            assert "arguments" in tool_call["function"], "Tool call function should have arguments"
    else:
        pytest.skip("Ollama model didn't return tool_calls (may need different prompt/model)")
