#!/usr/bin/env python3
"""
Minimal Bedrock reproduction script for debugging multi-tool-call validation errors.

This script bypasses pytest's module-scoped fixtures to create fresh BedrockClient
instances and includes comprehensive debug logging for Bedrock Converse API interactions.

Error reproduced:
    ValidationException: Expected toolResult blocks at messages.2.content for the following Ids: <tool_id>

Hypothesis:
    When Bedrock generates 2+ simultaneous tool calls, one tool result may appear
    missing due to message batching or formatting issues.

Prerequisites:
    - Valid AWS credentials configured (AWS_PROFILE or ~/.aws/credentials)
    - Access to Bedrock service in your AWS account
    - bedrock:InvokeModel permission for amazon.nova-pro-v1:0

Usage:
    uv run python debug_bedrock_repro.py

The script will:
    1. Create a fresh BedrockClient (bypassing pytest fixtures)
    2. Make an initial API call that triggers multiple tool calls
    3. Format tool results and make a second API call
    4. Log detailed request/response data at each step
    5. Catch and report the ValidationException if it occurs
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary modules from chatboti
from chatboti.llm import BedrockClient


def log_section(title: str):
    """Print a section header for clarity in debug output."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def log_messages(messages: List[Dict[str, Any]], title: str = "Messages"):
    """Pretty-print messages for debugging."""
    log_section(title)
    for i, msg in enumerate(messages):
        print(f"\n[{i}] Role: {msg.get('role', 'UNKNOWN')}")
        content = msg.get('content', '')

        if isinstance(content, list):
            print(f"    Content (list with {len(content)} items):")
            for j, item in enumerate(content):
                if isinstance(item, dict):
                    if 'text' in item:
                        text = item['text'][:200] + '...' if len(item['text']) > 200 else item['text']
                        print(f"      [{j}] text: {text}")
                    elif 'toolUse' in item:
                        tool_use = item['toolUse']
                        print(f"      [{j}] toolUse:")
                        print(f"          toolUseId: {tool_use.get('toolUseId')}")
                        print(f"          name: {tool_use.get('name')}")
                        print(f"          input: {json.dumps(tool_use.get('input', {}))}")
                    elif 'toolResult' in item:
                        tool_result = item['toolResult']
                        print(f"      [{j}] toolResult:")
                        print(f"          toolUseId: {tool_result.get('toolUseId')}")
                        print(f"          status: {tool_result.get('status')}")
                        content_text = tool_result.get('content', [{}])[0].get('text', '')
                        content_preview = content_text[:200] + '...' if len(content_text) > 200 else content_text
                        print(f"          content: {content_preview}")
                else:
                    print(f"      [{j}] {item}")
        else:
            text = str(content)[:200] + '...' if len(str(content)) > 200 else str(content)
            print(f"    Content: {text}")

        if 'tool_calls' in msg:
            print(f"    Tool calls: {len(msg['tool_calls'])}")
            for j, tc in enumerate(msg['tool_calls']):
                print(f"      [{j}] id: {tc.get('id', 'N/A')}")
                print(f"          function: {tc.get('function', {}).get('name')}")
                print(f"          arguments: {tc.get('function', {}).get('arguments')}")

        if 'tool_call_id' in msg:
            print(f"    Tool call ID: {msg['tool_call_id']}")
            print(f"    Status: {msg.get('status', 'N/A')}")


def log_api_request(request_kwargs: Dict[str, Any]):
    """Log the Bedrock Converse API request."""
    log_section("BEDROCK CONVERSE API REQUEST")
    print(f"\nModel ID: {request_kwargs.get('modelId')}")
    print(f"\nInference Config:")
    print(f"  {json.dumps(request_kwargs.get('inferenceConfig', {}), indent=2)}")

    if 'system' in request_kwargs:
        print(f"\nSystem blocks: {len(request_kwargs['system'])}")
        for i, block in enumerate(request_kwargs['system']):
            text = block.get('text', '')[:200] + '...' if len(block.get('text', '')) > 200 else block.get('text', '')
            print(f"  [{i}] {text}")

    print(f"\nMessages: {len(request_kwargs.get('messages', []))}")
    for i, msg in enumerate(request_kwargs.get('messages', [])):
        print(f"\n  Message [{i}] Role: {msg['role']}")
        content = msg.get('content', [])
        if isinstance(content, list):
            print(f"    Content blocks: {len(content)}")
            for j, block in enumerate(content):
                if 'text' in block:
                    text = block['text'][:100] + '...' if len(block['text']) > 100 else block['text']
                    print(f"      [{j}] text: {text}")
                elif 'toolUse' in block:
                    tool_use = block['toolUse']
                    print(f"      [{j}] toolUse: {tool_use.get('name')} (id: {tool_use.get('toolUseId')})")
                elif 'toolResult' in block:
                    tool_result = block['toolResult']
                    print(f"      [{j}] toolResult for id: {tool_result.get('toolUseId')} (status: {tool_result.get('status')})")

    if 'toolConfig' in request_kwargs:
        tools = request_kwargs['toolConfig'].get('tools', [])
        print(f"\nTools available: {len(tools)}")
        for i, tool in enumerate(tools):
            spec = tool.get('toolSpec', {})
            print(f"  [{i}] {spec.get('name')}: {spec.get('description', '')[:80]}...")


def log_api_response(response: Dict[str, Any]):
    """Log the Bedrock Converse API response."""
    log_section("BEDROCK CONVERSE API RESPONSE")

    print(f"\nStop Reason: {response.get('stopReason')}")
    print(f"\nUsage:")
    usage = response.get('usage', {})
    print(f"  Input tokens: {usage.get('inputTokens')}")
    print(f"  Output tokens: {usage.get('outputTokens')}")

    output = response.get('output', {})
    message = output.get('message', {})
    print(f"\nMessage Role: {message.get('role')}")

    content = message.get('content', [])
    print(f"Content blocks: {len(content)}")
    for i, block in enumerate(content):
        print(f"\n  Block [{i}]:")
        if 'text' in block:
            text = block['text'][:200] + '...' if len(block['text']) > 200 else block['text']
            print(f"    text: {text}")
        elif 'toolUse' in block:
            tool_use = block['toolUse']
            print(f"    toolUse:")
            print(f"      toolUseId: {tool_use.get('toolUseId')}")
            print(f"      name: {tool_use.get('name')}")
            print(f"      input: {json.dumps(tool_use.get('input', {}), indent=8)}")


def create_test_tools() -> List[Dict[str, Any]]:
    """Create tool definitions matching the test scenario."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_all_speakers",
                "description": "Get list of all available speaker names.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_matching_speakers",
                "description": "Find the top N most relevant speakers using semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Topic, technology, or expertise area"
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results (default: 3, max recommended: 5)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


def create_mock_tool_results(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create mock tool results for testing."""
    results = []

    for tc in tool_calls:
        tool_name = tc['function']['name']
        tool_call_id = tc['function'].get('tool_call_id', tc.get('id', ''))

        if tool_name == 'list_all_speakers':
            content = json.dumps({
                "success": True,
                "speakers": ["Dr. Jane Smith", "Prof. John Doe", "Dr. Alice Wong"],
                "intro_message": "Conference Speakers from the data"
            })
        elif tool_name == 'get_matching_speakers':
            args = json.loads(tc['function']['arguments'])
            content = json.dumps({
                "success": True,
                "speakers": [
                    {
                        "Name": "Dr. Jane Smith",
                        "Final title": "AI Research and Machine Learning",
                        "Bio": "Dr. Smith has 15 years of experience...",
                        "Abstract": "This talk explores recent advances..."
                    }
                ],
                "count": 1,
                "query": args.get('query', ''),
                "total_speakers_searched": 3
            })
        else:
            content = json.dumps({"success": False, "error": "Unknown tool"})

        results.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "status": "success"
        })

    return results


async def reproduce_bedrock_issue():
    """Reproduce the Bedrock multi-tool-call validation error."""
    log_section("BEDROCK REPRO SCRIPT STARTED")

    # Create fresh BedrockClient (bypasses pytest caching)
    logger.info("Creating fresh BedrockClient instance...")
    client = BedrockClient(model="amazon.nova-pro-v1:0")

    try:
        await client.connect()
        logger.info(f"Connected to Bedrock model: {client.model}")

        # Prepare tools
        tools = create_test_tools()
        log_section("TOOLS DEFINED")
        for i, tool in enumerate(tools):
            print(f"  [{i}] {tool['function']['name']}: {tool['function']['description']}")

        # Initial messages - simulate the failing test scenario
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can use tools to answer questions. "
                    "You MUST call the tools to gather information before answering. "
                    "Call list_all_speakers to count speakers, then get_matching_speakers to find relevant ones."
                )
            },
            {
                "role": "user",
                "content": (
                    "Use the list_all_speakers tool, then use get_matching_speakers to find a speaker for 'machine learning'. "
                    "Tell me how many speakers there are and who is the best speaker for 'machine learning'."
                )
            }
        ]

        log_messages(messages, "INITIAL MESSAGES (Before First API Call)")

        # First API call - expect tool calls
        log_section("FIRST API CALL")
        logger.info("Calling get_completion with tools...")
        response1 = await client.get_completion(messages, tools=tools, temperature=0.0)

        log_section("FIRST RESPONSE RECEIVED")
        print(f"\nResponse text: {response1.get('text', '')}")
        print(f"Tool calls: {len(response1.get('tool_calls', []))}")

        tool_calls = response1.get('tool_calls', [])
        if not tool_calls:
            logger.error("NO TOOL CALLS RETURNED - Test scenario not reproduced")
            return

        logger.info(f"Received {len(tool_calls)} tool calls")
        for i, tc in enumerate(tool_calls):
            logger.info(f"  [{i}] {tc['function']['name']} (id: {tc['function'].get('tool_call_id', 'N/A')})")

        # Build assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": response1.get('text') or "",
            "tool_calls": [
                {
                    "id": tc['function'].get('tool_call_id', tc.get('id', '')),
                    "type": "function",
                    "function": {
                        "name": tc['function']['name'],
                        "arguments": tc['function']['arguments']
                    }
                }
                for tc in tool_calls
            ]
        }
        messages.append(assistant_msg)

        # Create tool results
        tool_results = create_mock_tool_results(tool_calls)
        log_section("MOCK TOOL RESULTS CREATED")
        for i, result in enumerate(tool_results):
            preview = result['content'][:100] + '...' if len(result['content']) > 100 else result['content']
            logger.info(f"  [{i}] tool_call_id={result['tool_call_id']}, status={result['status']}")
            logger.info(f"       content={preview}")

        # Add all tool results to messages
        for result in tool_results:
            messages.append(result)

        log_messages(messages, "MESSAGES BEFORE SECOND API CALL (With Tool Results)")

        # This is where the error occurs - second API call with tool results
        log_section("SECOND API CALL (Expected to fail)")
        logger.info("Calling get_completion with tool results in message history...")

        # Log what the client will transform the messages into
        logger.info("Calling client._transform_messages to see Bedrock format...")
        request_kwargs = client._transform_messages(messages, tools)
        log_api_request(request_kwargs)

        try:
            response2 = await client.get_completion(messages, tools=tools, temperature=0.0)

            log_section("SECOND RESPONSE RECEIVED (No error!)")
            print(f"\nResponse text: {response2.get('text', '')}")
            print(f"Tool calls: {len(response2.get('tool_calls', []))}")

            logger.info("SUCCESS - No validation error occurred!")

        except Exception as e:
            log_section("ERROR OCCURRED (Issue Reproduced!)")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")

            if "ValidationException" in str(e) and "Expected toolResult blocks" in str(e):
                logger.error("✓ Successfully reproduced the Bedrock validation error!")
                logger.error("This confirms the issue with multi-tool-call handling.")
            else:
                logger.error("Different error occurred - may not be the target issue")

            raise

    finally:
        await client.close()
        logger.info("BedrockClient closed")


async def main():
    """Main entry point."""
    try:
        await reproduce_bedrock_issue()
        logger.info("Script completed successfully!")
        return 0
    except Exception as e:
        logger.exception("Reproduction script failed")

        # Provide helpful error messages for common issues
        error_msg = str(e)
        if "InvalidClientTokenId" in error_msg or "security token" in error_msg:
            logger.error("\n" + "=" * 80)
            logger.error("AWS CREDENTIALS ERROR")
            logger.error("=" * 80)
            logger.error("Please ensure you have valid AWS credentials configured:")
            logger.error("  1. Set AWS_PROFILE environment variable, or")
            logger.error("  2. Configure credentials with: aws configure")
            logger.error("  3. Ensure the credentials have Bedrock access")
        elif "AccessDenied" in error_msg:
            logger.error("\n" + "=" * 80)
            logger.error("AWS PERMISSIONS ERROR")
            logger.error("=" * 80)
            logger.error("Your AWS credentials don't have Bedrock access.")
            logger.error("Required permissions:")
            logger.error("  - bedrock:InvokeModel")
            logger.error("  - bedrock:GetFoundationModel (optional)")

        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
