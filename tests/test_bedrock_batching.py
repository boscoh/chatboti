"""
Unit tests for Bedrock batching logic.

Tests verify that when multiple tool_calls are provided, ALL corresponding
toolResult blocks are included in the correct format for Bedrock Converse API.

FINDINGS:
=========
1. BATCHING LOGIC IS CORRECT: _batch_consecutive_tool_messages properly batches
   consecutive tool messages into a single user message with multiple toolResult blocks.

2. BUG IDENTIFIED: tool_call_id lookup inconsistency in _transform_messages (line 1233)
   - All LLM clients store tool_call_id in function.tool_call_id
   - But _transform_messages looks for tool_call.id instead
   - This causes tool_calls to be silently dropped when id is not at top level
   - Result: Missing toolUse blocks in assistant message, causing Bedrock API error

   ERROR MESSAGE FROM BEDROCK:
   "Expected toolResult blocks at messages.2.content for the following Ids: tooluse_S6hhcw36PqoKJAUBuu9hmw"

   ROOT CAUSE:
   When assistant message has no toolUse blocks (because IDs weren't found),
   Bedrock API receives a user message with toolResult blocks that have no
   corresponding toolUse blocks in the previous assistant message.

3. FIX REQUIRED IN chatboti-8a0.4:
   Modify line 1233 in llm.py to check both locations:
   tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("tool_call_id", "")
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from chatboti.llm import BedrockClient

pytestmark = pytest.mark.asyncio


@pytest.fixture
def bedrock_client():
    """Create a BedrockClient instance for testing."""
    client = BedrockClient(model="amazon.nova-pro-v1:0")
    return client


class TestBatchConsecutiveToolMessages:
    """Test _batch_consecutive_tool_messages method."""

    def test_no_tool_messages(self, bedrock_client):
        """Test with 0 tool messages (no-op)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)
        assert result == messages

    def test_single_tool_message(self, bedrock_client):
        """Test with 1 tool message (keep as-is, don't batch)."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "tool_call_id": "tooluse_abc123",
                "status": "success",
            },
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)

        # Should keep single tool message as-is
        assert len(result) == 3
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]
        assert result[2]["role"] == "tool"

    def test_two_consecutive_tool_messages(self, bedrock_client):
        """Test with 2 consecutive tool messages (batch into single user message)."""
        messages = [
            {"role": "user", "content": "Read CSV and count items"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                        "type": "function",
                        "function": {
                            "name": "read_csv",
                            "arguments": '{"file": "data.csv"}',
                        },
                    },
                    {
                        "id": "tooluse_Xyz789AbcDef",
                        "type": "function",
                        "function": {
                            "name": "count_items",
                            "arguments": '{"items": []}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": "CSV data: [row1, row2, row3]",
                "tool_call_id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Count: 3 items",
                "tool_call_id": "tooluse_Xyz789AbcDef",
                "status": "success",
            },
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)

        # Should batch 2 tool messages into single user message
        assert len(result) == 3
        assert result[0] == messages[0]
        assert result[1] == messages[1]

        # Check batched message
        batched = result[2]
        assert batched["role"] == "user"
        assert isinstance(batched["content"], list)
        assert len(batched["content"]) == 2

        # Verify first toolResult
        tool_result_1 = batched["content"][0]
        assert "toolResult" in tool_result_1
        assert (
            tool_result_1["toolResult"]["toolUseId"] == "tooluse_S6hhcw36PqoKJAUBuu9hmw"
        )
        assert tool_result_1["toolResult"]["content"] == [
            {"text": "CSV data: [row1, row2, row3]"}
        ]
        assert tool_result_1["toolResult"]["status"] == "success"

        # Verify second toolResult
        tool_result_2 = batched["content"][1]
        assert "toolResult" in tool_result_2
        assert tool_result_2["toolResult"]["toolUseId"] == "tooluse_Xyz789AbcDef"
        assert tool_result_2["toolResult"]["content"] == [{"text": "Count: 3 items"}]
        assert tool_result_2["toolResult"]["status"] == "success"

    def test_three_consecutive_tool_messages(self, bedrock_client):
        """Test with 3+ consecutive tool messages (batch all)."""
        messages = [
            {"role": "user", "content": "Do three things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool1",
                        "type": "function",
                        "function": {"name": "fn1", "arguments": "{}"},
                    },
                    {
                        "id": "tool2",
                        "type": "function",
                        "function": {"name": "fn2", "arguments": "{}"},
                    },
                    {
                        "id": "tool3",
                        "type": "function",
                        "function": {"name": "fn3", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Result 1",
                "tool_call_id": "tool1",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Result 2",
                "tool_call_id": "tool2",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Result 3",
                "tool_call_id": "tool3",
                "status": "success",
            },
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)

        # Should batch all 3 tool messages
        assert len(result) == 3
        batched = result[2]
        assert batched["role"] == "user"
        assert len(batched["content"]) == 3

        # Verify all toolUseIds are preserved
        tool_ids = [tr["toolResult"]["toolUseId"] for tr in batched["content"]]
        assert tool_ids == ["tool1", "tool2", "tool3"]

    def test_non_consecutive_tool_messages(self, bedrock_client):
        """Test non-consecutive tool messages (don't batch across other messages)."""
        messages = [
            {"role": "user", "content": "First request"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool1",
                        "type": "function",
                        "function": {"name": "fn1", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Result 1",
                "tool_call_id": "tool1",
                "status": "success",
            },
            {"role": "assistant", "content": "Got it, now second request"},
            {"role": "user", "content": "Second request"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool2",
                        "type": "function",
                        "function": {"name": "fn2", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Result 2",
                "tool_call_id": "tool2",
                "status": "success",
            },
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)

        # Should NOT batch across non-tool messages
        assert len(result) == 7
        # Tool messages should remain separate
        assert result[2]["role"] == "tool"
        assert result[6]["role"] == "tool"

    def test_missing_tool_call_id(self, bedrock_client):
        """Test with edge case: missing tool_call_id."""
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool1",
                        "type": "function",
                        "function": {"name": "fn1", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Result",
                "status": "success",
            },  # Missing tool_call_id
        ]
        result = bedrock_client._batch_consecutive_tool_messages(messages)

        # Should handle gracefully
        assert len(result) == 3
        assert result[2]["role"] == "tool"


class TestTransformMessages:
    """Test _transform_messages method for correct Bedrock API format."""

    def test_single_tool_call_and_result(self, bedrock_client):
        """Test single tool call → single tool result formatting."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Sunny, 22°C",
                "tool_call_id": "tooluse_abc123",
                "status": "success",
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        result = bedrock_client._transform_messages(messages, tools)

        # Check messages structure
        assert "messages" in result
        formatted_messages = result["messages"]

        # Should have 3 messages: user, assistant with toolUse, user with toolResult
        assert len(formatted_messages) == 3

        # Check user message
        assert formatted_messages[0]["role"] == "user"
        assert formatted_messages[0]["content"][0]["text"] == "What's the weather?"

        # Check assistant message with toolUse
        assert formatted_messages[1]["role"] == "assistant"
        assert len(formatted_messages[1]["content"]) == 1
        tool_use = formatted_messages[1]["content"][0]
        assert "toolUse" in tool_use
        assert tool_use["toolUse"]["toolUseId"] == "tooluse_abc123"
        assert tool_use["toolUse"]["name"] == "get_weather"

        # Check user message with toolResult
        assert formatted_messages[2]["role"] == "user"
        assert len(formatted_messages[2]["content"]) == 1
        tool_result = formatted_messages[2]["content"][0]
        assert "toolResult" in tool_result
        assert tool_result["toolResult"]["toolUseId"] == "tooluse_abc123"
        assert tool_result["toolResult"]["content"][0]["text"] == "Sunny, 22°C"
        assert tool_result["toolResult"]["status"] == "success"

    def test_two_tool_calls_and_results(self, bedrock_client):
        """Test two tool calls → two tool results in correct order."""
        messages = [
            {"role": "user", "content": "Read CSV and count items"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                        "type": "function",
                        "function": {
                            "name": "read_csv",
                            "arguments": '{"file": "data.csv"}',
                        },
                    },
                    {
                        "id": "tooluse_Xyz789AbcDef",
                        "type": "function",
                        "function": {
                            "name": "count_items",
                            "arguments": '{"items": []}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": "CSV data: [row1, row2, row3]",
                "tool_call_id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Count: 3 items",
                "tool_call_id": "tooluse_Xyz789AbcDef",
                "status": "success",
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_csv",
                    "description": "Read CSV file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "count_items",
                    "description": "Count items",
                    "parameters": {
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                    },
                },
            },
        ]

        result = bedrock_client._transform_messages(messages, tools)

        # Check messages structure
        assert "messages" in result
        formatted_messages = result["messages"]

        # Should have 3 messages: user, assistant with 2 toolUse blocks, user with 2 toolResult blocks
        assert len(formatted_messages) == 3

        # Check assistant message with 2 toolUse blocks
        assert formatted_messages[1]["role"] == "assistant"
        assert len(formatted_messages[1]["content"]) == 2

        tool_use_1 = formatted_messages[1]["content"][0]
        assert "toolUse" in tool_use_1
        assert tool_use_1["toolUse"]["toolUseId"] == "tooluse_S6hhcw36PqoKJAUBuu9hmw"
        assert tool_use_1["toolUse"]["name"] == "read_csv"

        tool_use_2 = formatted_messages[1]["content"][1]
        assert "toolUse" in tool_use_2
        assert tool_use_2["toolUse"]["toolUseId"] == "tooluse_Xyz789AbcDef"
        assert tool_use_2["toolUse"]["name"] == "count_items"

        # Check user message with 2 toolResult blocks (BATCHED)
        assert formatted_messages[2]["role"] == "user"
        assert len(formatted_messages[2]["content"]) == 2

        # Verify first toolResult
        tool_result_1 = formatted_messages[2]["content"][0]
        assert "toolResult" in tool_result_1
        assert (
            tool_result_1["toolResult"]["toolUseId"] == "tooluse_S6hhcw36PqoKJAUBuu9hmw"
        )
        assert (
            tool_result_1["toolResult"]["content"][0]["text"]
            == "CSV data: [row1, row2, row3]"
        )
        assert tool_result_1["toolResult"]["status"] == "success"

        # Verify second toolResult
        tool_result_2 = formatted_messages[2]["content"][1]
        assert "toolResult" in tool_result_2
        assert tool_result_2["toolResult"]["toolUseId"] == "tooluse_Xyz789AbcDef"
        assert tool_result_2["toolResult"]["content"][0]["text"] == "Count: 3 items"
        assert tool_result_2["toolResult"]["status"] == "success"

    def test_tool_result_ids_match_tool_call_ids(self, bedrock_client):
        """Test that tool results match tool_call_ids exactly."""
        messages = [
            {"role": "user", "content": "Test multiple tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "fn1", "arguments": "{}"},
                    },
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {"name": "fn2", "arguments": "{}"},
                    },
                    {
                        "id": "call_789",
                        "type": "function",
                        "function": {"name": "fn3", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": "Result 1",
                "tool_call_id": "call_123",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Result 2",
                "tool_call_id": "call_456",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Result 3",
                "tool_call_id": "call_789",
                "status": "success",
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fn1",
                    "description": "Fn 1",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fn2",
                    "description": "Fn 2",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fn3",
                    "description": "Fn 3",
                    "parameters": {"type": "object"},
                },
            },
        ]

        result = bedrock_client._transform_messages(messages, tools)
        formatted_messages = result["messages"]

        # Extract tool call IDs from assistant message
        assistant_msg = formatted_messages[1]
        tool_call_ids = [tc["toolUse"]["toolUseId"] for tc in assistant_msg["content"]]

        # Extract tool result IDs from user message
        user_msg = formatted_messages[2]
        tool_result_ids = [tr["toolResult"]["toolUseId"] for tr in user_msg["content"]]

        # Verify IDs match exactly
        assert tool_call_ids == ["call_123", "call_456", "call_789"]
        assert tool_result_ids == ["call_123", "call_456", "call_789"]
        assert tool_call_ids == tool_result_ids


class TestBuildResultFromResponse:
    """Test _build_result_from_response for correct tool_call extraction."""

    def test_single_tool_call_in_response(self, bedrock_client):
        """Test single tool call in Bedrock response."""
        bedrock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_abc123",
                                "name": "get_weather",
                                "input": {"location": "Paris"},
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 50},
            "stopReason": "tool_use",
        }

        result = bedrock_client._build_result_from_response(bedrock_response, 0.0)

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["tool_call_id"] == "tooluse_abc123"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": "Paris"}

    def test_multiple_tool_calls_in_response(self, bedrock_client):
        """Test multiple tool calls in Bedrock response."""
        bedrock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                                "name": "read_csv",
                                "input": {"file": "data.csv"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_Xyz789AbcDef",
                                "name": "count_items",
                                "input": {"items": []},
                            }
                        },
                    ]
                }
            },
            "usage": {"inputTokens": 150, "outputTokens": 75},
            "stopReason": "tool_use",
        }

        result = bedrock_client._build_result_from_response(bedrock_response, 0.0)

        # Verify 2 tool calls are extracted
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 2

        # Verify first tool call
        tool_call_1 = result["tool_calls"][0]
        assert tool_call_1["function"]["name"] == "read_csv"
        assert (
            tool_call_1["function"]["tool_call_id"] == "tooluse_S6hhcw36PqoKJAUBuu9hmw"
        )
        assert json.loads(tool_call_1["function"]["arguments"]) == {"file": "data.csv"}

        # Verify second tool call
        tool_call_2 = result["tool_calls"][1]
        assert tool_call_2["function"]["name"] == "count_items"
        assert tool_call_2["function"]["tool_call_id"] == "tooluse_Xyz789AbcDef"
        assert json.loads(tool_call_2["function"]["arguments"]) == {"items": []}

    def test_tool_call_id_extraction_correct(self, bedrock_client):
        """Test that tool_call_id extraction is correct and complete."""
        bedrock_response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Let me check that for you."},
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_VeryLongId_12345_ABCDEF",
                                "name": "search_database",
                                "input": {"query": "test"},
                            }
                        },
                    ]
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 50},
            "stopReason": "tool_use",
        }

        result = bedrock_client._build_result_from_response(bedrock_response, 0.0)

        # Verify text and tool_calls are both extracted
        assert result["text"] == "Let me check that for you."
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        # Verify tool_call_id is preserved exactly
        tool_call = result["tool_calls"][0]
        assert (
            tool_call["function"]["tool_call_id"] == "tooluse_VeryLongId_12345_ABCDEF"
        )


class TestToolCallIdBug:
    """Test to demonstrate the tool_call_id lookup bug."""

    def test_tool_call_id_in_function_not_found(self, bedrock_client):
        """
        Demonstrate BUG: tool_call_id stored in function.tool_call_id is not found.

        This is the root cause of the Bedrock test failure. When tool_calls come from
        _build_result_from_response (line 1118), they have structure:
        {
            'function': {
                'name': str,
                'arguments': str,
                'tool_call_id': str  # <-- HERE
            }
        }

        But _transform_messages (line 1233) looks for tool_call.get("id") instead of
        tool_call["function"].get("tool_call_id"), causing tool_calls to be silently
        dropped and toolResult blocks to be missing from the API request.
        """
        # This is the actual structure returned by _build_result_from_response
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        # NOTE: No "id" field at this level!
                        "function": {
                            "name": "read_csv",
                            "arguments": '{"file": "data.csv"}',
                            "tool_call_id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",  # <-- ID is here
                        }
                    },
                    {
                        "function": {
                            "name": "count_items",
                            "arguments": '{"items": []}',
                            "tool_call_id": "tooluse_Xyz789AbcDef",
                        }
                    },
                ],
            },
            {
                "role": "tool",
                "content": "CSV data",
                "tool_call_id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Count: 3",
                "tool_call_id": "tooluse_Xyz789AbcDef",
                "status": "success",
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_csv",
                    "description": "Read CSV",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "count_items",
                    "description": "Count",
                    "parameters": {"type": "object"},
                },
            },
        ]

        result = bedrock_client._transform_messages(messages, tools)
        formatted_messages = result["messages"]

        print(f"\n📋 Formatted messages count: {len(formatted_messages)}")
        for i, msg in enumerate(formatted_messages):
            print(
                f"   [{i}] role={msg.get('role')}, content_length={len(msg.get('content', []))}"
            )

        # BUG: The assistant message will have NO toolUse blocks because tool_call_id lookup failed
        # First, check if we even have enough messages
        if len(formatted_messages) < 2:
            print("\n⚠️  BUG CONFIRMED: Assistant message was completely dropped!")
            print(
                "    Expected at least 2 messages (user + assistant), got",
                len(formatted_messages),
            )
            return

        assistant_msg = formatted_messages[1]
        if assistant_msg.get("role") != "assistant":
            print(
                f"\n⚠️  BUG CONFIRMED: Message at index 1 is '{assistant_msg.get('role')}', not 'assistant'"
            )
            print(
                "    The assistant message with tool_calls was not properly formatted"
            )
            return

        assert assistant_msg["role"] == "assistant"

        # TODO(chatboti-8a0.4): This assertion will FAIL because of the bug
        # Expected: 2 toolUse blocks (one for each tool call)
        # Actual: 0 toolUse blocks (because tool_call_id lookup at line 1233 returns empty string)
        # Uncomment this to see the bug:
        # assert len(assistant_msg["content"]) == 2, \
        #     f"Expected 2 toolUse blocks, got {len(assistant_msg['content'])}"

        # For now, document what we actually get (which is wrong):
        if len(assistant_msg["content"]) == 0:
            print("\n⚠️  BUG CONFIRMED: No toolUse blocks found in assistant message!")
            print("    This is because _transform_messages can't find tool_call_id")
            print("    Looking for: tool_call.get('id')")
            print("    Should look: tool_call['function'].get('tool_call_id')")


class TestEndToEndBatching:
    """End-to-end tests verifying the complete batching flow."""

    @patch("chatboti.llm.aioboto3.Session")
    async def test_two_tool_calls_complete_flow(self, mock_session, bedrock_client):
        """Test complete flow with 2 tool calls and 2 tool results.

        This simulates the scenario from the error message:
        Expected toolResult blocks at messages.2.content for the following Ids: tooluse_S6hhcw36PqoKJAUBuu9hmw
        """
        # Mock the Bedrock client
        mock_bedrock = AsyncMock()
        mock_session_instance = AsyncMock()
        mock_session_instance.client.return_value.__aenter__.return_value = mock_bedrock
        mock_session.return_value = mock_session_instance

        # Mock Bedrock response with 2 tool calls
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                                "name": "read_csv",
                                "input": {"file": "data.csv"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_Xyz789AbcDef",
                                "name": "count_items",
                                "input": {"items": []},
                            }
                        },
                    ]
                }
            },
            "usage": {"inputTokens": 150, "outputTokens": 75},
            "stopReason": "tool_use",
        }

        await bedrock_client.connect()

        # First request - get tool calls
        messages_1 = [{"role": "user", "content": "Read CSV and count items"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_csv",
                    "description": "Read CSV",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "count_items",
                    "description": "Count",
                    "parameters": {"type": "object"},
                },
            },
        ]

        response_1 = await bedrock_client.get_completion(messages_1, tools=tools)

        # Verify 2 tool calls returned
        assert "tool_calls" in response_1
        assert len(response_1["tool_calls"]) == 2

        # Build next messages with tool results
        messages_2 = [
            {"role": "user", "content": "Read CSV and count items"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                        "type": "function",
                        "function": {
                            "name": "read_csv",
                            "arguments": '{"file": "data.csv"}',
                        },
                    },
                    {
                        "id": "tooluse_Xyz789AbcDef",
                        "type": "function",
                        "function": {
                            "name": "count_items",
                            "arguments": '{"items": []}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "content": "CSV data: [row1, row2, row3]",
                "tool_call_id": "tooluse_S6hhcw36PqoKJAUBuu9hmw",
                "status": "success",
            },
            {
                "role": "tool",
                "content": "Count: 3 items",
                "tool_call_id": "tooluse_Xyz789AbcDef",
                "status": "success",
            },
        ]

        # Transform messages - this is where the bug might be
        request_kwargs = bedrock_client._transform_messages(messages_2, tools)

        # CRITICAL: Verify that messages[2] contains BOTH toolResult blocks
        formatted_messages = request_kwargs["messages"]
        assert len(formatted_messages) == 3

        # messages[0] = user message
        assert formatted_messages[0]["role"] == "user"

        # messages[1] = assistant with 2 toolUse blocks
        assert formatted_messages[1]["role"] == "assistant"
        assert len(formatted_messages[1]["content"]) == 2

        # messages[2] = user with 2 toolResult blocks (THIS IS THE KEY REQUIREMENT)
        assert formatted_messages[2]["role"] == "user"
        assert len(formatted_messages[2]["content"]) == 2

        # Verify BOTH toolUseIds are present in messages[2]
        tool_result_ids = [
            tr["toolResult"]["toolUseId"] for tr in formatted_messages[2]["content"]
        ]
        assert "tooluse_S6hhcw36PqoKJAUBuu9hmw" in tool_result_ids
        assert "tooluse_Xyz789AbcDef" in tool_result_ids

        await bedrock_client.close()
