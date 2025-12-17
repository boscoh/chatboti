#!/usr/bin/env python3
"""
Test client demonstrating MCP tool integration with multi-step reasoning for speaker queries.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from microeval.chat_client import IChatClient, get_chat_client, load_config
from path import Path

model_config = load_config()
chat_models = model_config["chat_models"]

load_dotenv()

logger = logging.getLogger(__name__)


class SpeakerMcpClient:
    def __init__(self, chat_service: Optional[str] = None):
        self.chat_service = chat_service or os.getenv("CHAT_SERVICE")
        if not self.chat_service:
            raise ValueError("CHAT_SERVICE environment variable is not set")
        self._mcp_session: Optional[ClientSession] = None
        self._session_context: Optional[ClientSession] = None
        self._stdio_context: Optional[StdioServerParameters] = None

        self.tools: Optional[List[Dict[str, Any]]] = None

        self.chat_client: Optional[IChatClient] = None
        model = chat_models.get(self.chat_service)
        if not model:
            raise ValueError(f"Unsupported chat service: {self.chat_service}")
        self.chat_client = get_chat_client(self.chat_service, model=model)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def connect(self):
        if self._mcp_session:
            return

        env = os.environ.copy()
        server_script_path = Path(__file__).parent / "mcp_server.py"
        env["PYTHONPATH"] = server_script_path.parent
        env["CHAT_SERVICE"] = self.chat_service
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", server_script_path],
            env=env,
        )
        self._stdio_context = stdio_client(server_params)
        _stdio_read, _stdio_write = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(_stdio_read, _stdio_write)
        self._mcp_session = await self._session_context.__aenter__()
        await self._mcp_session.initialize()

        self.tools = await self.get_tools()
        names = [tool["function"]["name"] for tool in self.tools]
        logger.info(f"Connected Server to MCP tools: {', '.join(names)}")

        await self.chat_client.connect()

    async def disconnect(self):
        try:
            if self.chat_client:
                await self.chat_client.close()
        except Exception:
            pass
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            pass
        self._mcp_session = None

    async def get_tools(self):
        """Returns tool in format compatible with IChatClient.get_completion()"""
        response = await self._mcp_session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

    def parse_tool_args(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        tool_args_json = tool_call["function"].get("arguments", "")
        if not tool_args_json:
            return {}
        try:
            return json.loads(tool_args_json)
        except json.JSONDecodeError:
            return {"__raw": tool_args_json}

    def is_duplicate_call(
        self, tool_name: str, tool_args: Dict[str, Any], seen_calls: set
    ) -> bool:
        """Return True if this tool call was seen before; otherwise record it and return False."""
        try:
            normalized_args = json.dumps(tool_args, sort_keys=True)
        except Exception:
            normalized_args = str(tool_args)
        call_key = (tool_name, normalized_args)
        if call_key in seen_calls:
            logger.info(f"Skipped duplicate tool call: {call_key}")
            return True
        seen_calls.add(call_key)
        return False

    def _extract_content_text(self, item: Any) -> str:
        """Extract text representation from a message content item."""
        if isinstance(item, dict):
            if "text" in item:
                return item["text"]
            elif "toolUse" in item:
                return f"[toolUse: {item['toolUse'].get('name', 'unknown')}]"
            elif "toolResult" in item:
                return f"[toolResult: {item['toolResult'].get('toolUseId', 'unknown')}]"
        return str(item)

    def log_messages(self, messages: List[Dict[str, Any]], max_length: int = 100):
        """Log each message with truncated content if too long."""
        logger.info(f"Calling LLM with {len(messages)} messages:")
        for msg in messages:
            msg_content = msg.get("content", "")

            if isinstance(msg_content, list):
                content_parts = [
                    self._extract_content_text(item) for item in msg_content
                ]
                content_str = " ".join(content_parts)
            else:
                content_str = str(msg_content)

            content_str = content_str.replace("\r", "")
            content_str = re.sub(r"\s+", " ", content_str).strip()

            truncated_content = content_str[:max_length] + (
                "..." if len(content_str) > max_length else ""
            )
            role = msg.get("role", "unknown")
            logger.info(f"- {role}: {truncated_content}")

    async def process_query(
        self, query: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Returns a response to a user query by getting a completion with tool calls.

        Supports multi-step tool chaining (e.g., find correct name -> get_speaker_by_name)
        by iteratively executing returned tool calls and re-querying the model
        with tool outputs until no more tool calls are requested or a safety
        limit is reached.

        Args:
            query: The user's query string
            history: Optional list of previous messages with 'role' and 'content' keys
        """
        await self.connect()

        system_prompt = """You are a helpful assistant that can use tools to answer 
        questions about speakers.

        IMPORTANT: Proactively use tools in multiple rounds to gather, refine, and
        verify information. Prefer taking several small, iterative tool steps over
        guessing. You should:

        MULTI-ROUND TOOL CHAINING STRATEGY:
        1. ANALYZE the query and list what you need to know to answer it well
        2. PLAN a sequence of tool calls (potentially across multiple rounds)
        3. EXECUTE one or more tool calls, then reassess what you learned
        4. ITERATE with additional calls to fill gaps, cross-check, or drill down
        5. AVOID exact duplicate calls with identical parameters (vary params to explore)
        6. SYNTHESIZE the gathered evidence into a comprehensive final answer

        TOOL USAGE GUIDELINES:
        - Feel free to make multiple tool calls across multiple reasoning rounds
        - Use follow-up calls to verify, compare alternatives, and resolve ambiguity
        - Avoid repeating the exact same call with the same parameters
        - ALWAYS provide a complete, detailed final answer that includes specific
          information you obtained via the tools

        Available tools can help you search, filter, and retrieve speaker
        information. Use them iteratively and transparently. Your final answer
        should be detailed and include specific speaker information."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if history:
            for msg in history:
                role = msg.get("role", "user")
                if role in ("user", "assistant", "tool"):
                    messages.append(msg)

        messages.append({"role": "user", "content": str(query)})

        self.log_messages(messages)

        response = await self.chat_client.get_completion(messages, self.tools)

        tool_calls = response.get("tool_calls")
        max_iterations = 5
        iterations = 0
        seen_calls = set()

        while tool_calls and iterations < max_iterations:
            iterations += 1
            logger.info(
                f"Reasoning step {iterations} with {len(tool_calls)} tool calls"
            )

            if tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("text") or None,
                    "tool_calls": [
                        {
                            "id": tool_call["function"].get("tool_call_id")
                            or tool_call["function"].get("toolUseId", ""),
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"].get(
                                    "arguments",
                                    json.dumps(self.parse_tool_args(tool_call)),
                                ),
                            },
                        }
                        for tool_call in tool_calls
                        if tool_call["function"].get("tool_call_id")
                        or tool_call["function"].get("toolUseId")
                    ],
                }
                if assistant_msg["tool_calls"]:
                    messages.append(assistant_msg)

                tool_result_messages = []
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = self.parse_tool_args(tool_call)
                    tool_call_id = tool_call["function"].get(
                        "tool_call_id"
                    ) or tool_call["function"].get("toolUseId", "")

                    if not tool_call_id:
                        logger.warning(
                            f"Skipping tool call {tool_name} without tool_call_id"
                        )
                        continue

                    if self.is_duplicate_call(tool_name, tool_args, seen_calls):
                        result_content = (
                            f"Duplicate tool call: {tool_name}({tool_args})"
                        )
                        status = "error"
                    else:
                        try:
                            logger.info(f"Calling tool {tool_name}({tool_args})...")
                            result = await self._mcp_session.call_tool(
                                tool_name, tool_args
                            )
                            result_content = str(getattr(result, "content", result))
                            status = "success"
                        except Exception as e:
                            result_content = f"Tool {tool_name} failed: {str(e)}"
                            status = "error"
                            logger.error(f"Tool {tool_name} error: {e}")

                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "content": result_content,
                            "tool_call_id": tool_call_id,
                            "status": status,
                        }
                    )

                messages.extend(tool_result_messages)
            elif content := response.get("text", ""):
                messages.append({"role": "assistant", "content": content})

            self.log_messages(messages)

            response = await self.chat_client.get_completion(messages, self.tools)

            tool_calls = response.get("tool_calls")

        if iterations >= max_iterations:
            logger.warning(f"Reached maximum tool iterations ({max_iterations})")

        return response.get("text", "")


async def setup_async_exception_handler():
    loop = asyncio.get_running_loop()

    def silence_event_loop_closed(loop, context):
        if "exception" not in context or not isinstance(
            context["exception"], (RuntimeError, GeneratorExit)
        ):
            loop.default_exception_handler(context)

    loop.set_exception_handler(silence_event_loop_closed)


async def amain(service):
    await setup_async_exception_handler()
    async with SpeakerMcpClient(service) as client:
        for tool in client.tools:
            logger.info("----------------------------------------------")
            logger.info(f"Tool: {tool['function']['name']}")
            logger.info("Description:")
            for line in tool["function"]["description"].split("\n"):
                logger.info(f"| {line}")
        logger.info("----------------------------------------------")
        print("Type your query to pick a speaker.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        conversation_history: List[Dict[str, Any]] = []
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q", ""]:
                print("Goodbye!")
                return
            response = await client.process_query(
                query=user_input, history=conversation_history
            )
            print(f"\nResponse: {response}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
