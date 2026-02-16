#!/usr/bin/env python3
"""MCP-based agent with multi-step tool chaining."""

import asyncio
import json
import logging
import os
import re
import textwrap
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from microeval.llm import SimpleLLMClient, get_llm_client, load_config
from path import Path
import pydash as py_

from chatboti.config import get_chat_client

model_config = load_config()
chat_models = model_config["chat_models"]

load_dotenv()

logger = logging.getLogger(__name__)


class InfoAgent:
    def __init__(self, chat_client: SimpleLLMClient):
        """Initialize InfoAgent with a chat client.

        :param chat_client: Connected chat client (required)
        """
        self.chat_client = chat_client
        self.chat_service = getattr(chat_client, 'service', 'unknown')

        self._mcp_session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

        self.tools: Optional[List[Dict[str, Any]]] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False

    async def connect(self):
        if self._mcp_session:
            return

        self._exit_stack = AsyncExitStack()

        env = os.environ.copy()
        env["CHAT_SERVICE"] = self.chat_service

        uv_command = os.path.expanduser("~/.local/bin/uv")
        if not os.path.exists(uv_command):
            uv_command = "uv"

        server_params = StdioServerParameters(
            command=uv_command,
            args=["run", "-m", "chatboti.mcp_server"],
            env=env,
        )

        try:
            logger.info(f"Starting MCP stdio client with command: {uv_command}")
            _stdio_read, _stdio_write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            logger.info("MCP stdio connection established")

            self._mcp_session = await self._exit_stack.enter_async_context(
                ClientSession(_stdio_read, _stdio_write)
            )
            logger.info("MCP session created, initializing...")

            await self._mcp_session.initialize()
            logger.info("MCP session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            logger.error(
                "MCP initialization failed. Common issues:\n"
                "  1. MCP server subprocess failed to start\n"
                "  2. Missing environment variables (CHAT_SERVICE, AWS_REGION)\n"
                "  3. AWS credentials or IAM role issues\n"
                "  4. Dependency errors in MCP server\n"
                "  5. Port already in use"
            )
            raise

        self.tools = await self.get_tools()
        names = [py_.get(tool, "function.name", "") for tool in self.tools]
        logger.info(f"Connected Server to MCP tools: {', '.join(names)}")

        try:
            logger.info(f"Connecting chat client for service: {self.chat_service}")
            await self.chat_client.connect()
            logger.info("Chat client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect chat client: {e}")
            raise

    async def disconnect(self):
        try:
            if self.chat_client:
                await self.chat_client.close()
        except Exception:
            pass

        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing exit stack: {e}")

        self._mcp_session = None
        self._exit_stack = None

    async def get_tools(self):
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

    def _normalize_tool_args_to_json_str(self, tool_args: Any) -> str:
        """
        :param tool_args: Arguments as dict, string, or other type
        :return: Normalized JSON string with sorted keys
        """
        if not tool_args:
            return "{}"

        if isinstance(tool_args, dict):
            try:
                return json.dumps(tool_args, sort_keys=True)
            except Exception as e:
                logger.warning(f"Failed to serialize dict to JSON: {e}")
                return "{}"

        if isinstance(tool_args, str):
            try:
                parsed = json.loads(tool_args)
                return json.dumps(parsed, sort_keys=True)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON string: {tool_args[:100]}... Error: {e}")
                return "{}"

        logger.warning(f"Unexpected tool arguments type: {type(tool_args)}, using empty dict")
        return "{}"

    def _parse_tool_args(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool arguments from JSON string or dict.

        :param tool_call: Tool call dict with function.arguments
        :return: Parsed arguments dict, or empty dict if invalid
        """
        tool_args = py_.get(tool_call, "function.arguments", "")
        if not tool_args:
            return {}

        if isinstance(tool_args, dict):
            return tool_args

        if isinstance(tool_args, str):
            try:
                return json.loads(tool_args)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse tool arguments as JSON: {e}. "
                    f"Raw: {tool_args[:100]}{'...' if len(tool_args) > 100 else ''}"
                )
                return {}

        logger.warning(f"Unexpected tool arguments type: {type(tool_args)}")
        return {}

    def _is_duplicate_call(
        self, tool_name: str, tool_args: Dict[str, Any], seen_calls: set
    ) -> bool:
        """Check if a tool call is a duplicate.

        :param tool_name: Name of the tool
        :param tool_args: Tool arguments as dict
        :param seen_calls: Set of seen (tool_name, normalized_args) tuples
        :return: True if duplicate, False otherwise
        """
        args_str = self._normalize_tool_args_to_json_str(tool_args)
        call_key = (tool_name, args_str)
        if call_key in seen_calls:
            logger.info(f"Skipped duplicate tool call: {call_key}")
            return True
        seen_calls.add(call_key)
        return False

    def _extract_content_text(self, item: Any) -> str:
        return str(py_.get(item, "text", item))

    def _log_messages(self, messages: List[Dict[str, Any]], max_length: int = 100):
        logger.info(f"Calling LLM with {len(messages)} messages:")
        for msg in messages:
            content = py_.get(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(self._extract_content_text(item) for item in content)
            content = re.sub(r"\s+", " ", str(content).replace("\r", "")).strip()
            truncated = content[:max_length] + ("..." if len(content) > max_length else "")
            logger.info(f"- {py_.get(msg, 'role', 'unknown')}: {truncated}")

    def _get_tool_call_id(self, tool_call: Dict[str, Any]) -> str:
        return py_.get(tool_call, "function.tool_call_id", "")

    def _sanitize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool call arguments to valid JSON string with sorted keys.

        :param tool_call: Tool call dict with function.arguments
        :return: Tool call with normalized function.arguments as JSON string
        """
        result = py_.clone_deep(tool_call)
        tool_args = py_.get(tool_call, "function.arguments", "")
        args_str = self._normalize_tool_args_to_json_str(tool_args)
        return py_.set_(result, "function.arguments", args_str)

    def _sanitize_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        result = py_.clone_deep(msg)
        if tool_calls := py_.get(result, "tool_calls", []):
            py_.set_(result, "tool_calls", py_.map_(tool_calls, self._sanitize_tool_call))
        return result

    def _build_assistant_message(
        self, response: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        formatted_calls = []
        for tc in tool_calls:
            call_id = self._get_tool_call_id(tc)
            if not call_id:
                continue

            sanitized = self._sanitize_tool_call(tc)
            sanitized = py_.set_(sanitized, "id", call_id)
            sanitized = py_.set_(sanitized, "type", py_.get(sanitized, "type", "function"))
            formatted_calls.append(sanitized)

        if not formatted_calls:
            return None

        return {
            "role": "assistant",
            "content": py_.get(response, "text", None),
            "tool_calls": formatted_calls,
        }

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a helpful assistant that can use tools to answer questions about speakers.

        CRITICAL: You MUST call tools directly - NEVER explain what tool to call or describe the format.
        ALWAYS use the actual tool call mechanism - DO NOT write text describing what you would call.

        IMPORTANT: Use get_matching_speakers LIBERALLY with the 'n' parameter to gather comprehensive information.

        USING THE 'n' PARAMETER:
        - For comparisons: get_matching_speakers(query, n=3-5) to see multiple options
        - For comprehensive answers: Use n=3 to get diverse perspectives
        - For focused answers: Use n=1 for the single best match
        - ALWAYS prefer n=3+ for "compare", "all", "best" queries

        EXPLORATION STRATEGY FOR COMPLEX QUERIES:
        1. BREAK DOWN the query into multiple search angles
           - If asked about "AI and cloud", search for BOTH "AI" AND "cloud" separately
           - If comparing topics, search for EACH topic individually
           - If asked about multiple technologies, make separate calls for each

        2. SEARCH VARIATIONS - Try different query formulations:
           - Broad queries: "machine learning", "AI", "artificial intelligence"
           - Specific queries: "neural networks", "deep learning", "LLMs"
           - Related queries: "data science", "Python for ML"

        3. GATHER DIVERSE PERSPECTIVES:
           - Call get_matching_speaker_talk_and_bio multiple times with different queries
           - Compare results from different searches
           - Look for complementary expertise across speakers

        4. DON'T BE SHY ABOUT MULTIPLE CALLS:
           - It's BETTER to make 5-10 focused searches than 1 vague search
           - Each call gives you different information
           - More data = better, more comprehensive answer

        5. SYNTHESIZE ALL FINDINGS into a complete answer

        EXAMPLES:
        - Query: "AI and cloud speakers" → get_matching_speakers("AI", n=3), get_matching_speakers("cloud", n=3)
        - Query: "compare Python vs JavaScript" → get_matching_speakers("Python", n=3), get_matching_speakers("JavaScript", n=3)
        - Query: "who can speak about testing?" → get_matching_speakers("testing", n=5) to see all options
        - Query: "best speaker for beginners in AI" → get_matching_speakers("AI beginner friendly", n=3)

        REMEMBER: Making 10 tool calls is ENCOURAGED for thorough answers!

        NEVER SAY: "I would call X tool" or "To find Y, I would use Z"
        ALWAYS DO: Actually call the tool immediately - take action, don't describe it!"""
    )

    MAX_TOOL_ITERATIONS = 5

    async def _execute_tool(
        self, tool_call: Dict[str, Any], seen_calls: set
    ) -> Optional[Dict[str, Any]]:
        tool_name = py_.get(tool_call, "function.name", "")
        tool_args = self._parse_tool_args(tool_call)
        tool_call_id = self._get_tool_call_id(tool_call)

        if not tool_call_id:
            logger.warning(f"Skipping tool call {tool_name} without tool_call_id")
            return None

        result = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "status": "error",
            "content": "",
        }

        if self._is_duplicate_call(tool_name, tool_args, seen_calls):
            result["content"] = f"Duplicate tool call: {tool_name}({tool_args})"
        else:
            try:
                logger.info(f"Calling tool {tool_name}({tool_args})...")
                tool_result = await self._mcp_session.call_tool(tool_name, tool_args)
                result["content"] = str(getattr(tool_result, "content", tool_result))
                result["status"] = "success"
            except Exception as e:
                logger.error(f"Tool {tool_name} error: {e}")
                result["content"] = f"Tool {tool_name} failed: {str(e)}"

        return result

    def _build_initial_messages(
        self, query: str, history: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if history:
            for msg in history:
                if py_.get(msg, "role", "") in ("user", "assistant", "tool"):
                    messages.append(self._sanitize_message(msg))

        # Only add the query if it's not already the last user message in history
        should_add_query = True
        if messages:
            last_msg = messages[-1]
            if py_.get(last_msg, "role", "") == "user" and py_.get(last_msg, "content", "") == str(query):
                should_add_query = False

        if should_add_query:
            messages.append({"role": "user", "content": str(query)})

        return messages

    async def process_query(
        self, query: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Process user query with multi-step tool chaining.

        :param query: User's query string
        :param history: Optional conversation history
        :return: Final LLM response text
        """
        await self.connect()

        messages = self._build_initial_messages(query, history)
        self._log_messages(messages)

        response = await self.chat_client.get_completion(messages, self.tools)
        tool_calls = py_.get(response, "tool_calls", None)
        seen_calls: set = set()

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            if not tool_calls:
                break

            logger.info(f"Reasoning step {iteration + 1} with {len(tool_calls)} tool calls")

            assistant_msg = self._build_assistant_message(response, tool_calls)
            if assistant_msg:
                messages.append(assistant_msg)

            for tc in tool_calls:
                result_msg = await self._execute_tool(tc, seen_calls)
                if result_msg:
                    messages.append(result_msg)

            self._log_messages(messages)
            response = await self.chat_client.get_completion(messages, self.tools)
            tool_calls = py_.get(response, "tool_calls", None)
        else:
            if tool_calls:
                logger.warning(f"Reached maximum tool iterations ({self.MAX_TOOL_ITERATIONS})")

        return py_.get(response, "text", "")


async def setup_async_exception_handler():
    loop = asyncio.get_running_loop()

    def silence_event_loop_closed(loop, context):
        if "exception" not in context or not isinstance(
            context["exception"], (RuntimeError, GeneratorExit)
        ):
            loop.default_exception_handler(context)

    loop.set_exception_handler(silence_event_loop_closed)


async def amain():
    await setup_async_exception_handler()
    chat_client = await get_chat_client()
    try:
        async with InfoAgent(chat_client) as agent:
            client = agent
            for tool in client.tools:
                logger.info("----------------------------------------------")
                logger.info(f"Tool: {py_.get(tool, 'function.name')}")
                logger.info("Description:")
                for line in py_.get(tool, "function.description", "").split("\n"):
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
    finally:
        await chat_client.close()

