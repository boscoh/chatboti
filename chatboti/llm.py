"""
Simple chat client abstraction for LLM providers.

- Async only, with proper async context management
- No streaming, only conversation with tools and embeddings
- Mostly OpenAI JSON structure, but without choices, and easier token usage metadata
- No langchain, litellm etc., just vendor-provided Python packages
"""

# Standard library
import asyncio
import configparser
import copy
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

# Third-party
import aioboto3
import boto3
import groq
import ollama
import openai
from botocore.exceptions import ClientError, ProfileNotFound

logger = logging.getLogger(__name__)

# Logging strategy:
# - logger.error(): Exceptions returned as errors to caller
# - logger.warning(): Recoverable issues with fallback (missing config, expired tokens, unknown models)
# - logger.info(): Normal operations (initialization, config loading)
# - logger.debug(): Diagnostic details (message transformations, batching)

# Delay needed for aioboto3 to properly cleanup async resources
AIOBOTO3_CLEANUP_DELAY_SECONDS = 0.1

LLMService = Literal["openai", "ollama", "bedrock", "groq"]


def load_config() -> Dict[str, Any]:
    """
    Load and return the models configuration from models.json.

    Returns:
        Dict[str, Any]: Configuration dictionary with chat_models and embed_models
    """
    config_path = Path(__file__).parent / "models.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded selectable models from '{config_path}'")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using fallback config")
        # Fallback config if file doesn't exist
        return {
            "chat_models": {
                "bedrock": ["amazon.nova-pro-v1:0"],
                "openai": ["gpt-4o"],
                "ollama": ["llama3.2"],
                "groq": ["llama-3.3-70b-versatile"],
            },
            "embed_models": {
                "openai": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                "ollama": ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
                "bedrock": ["amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v1", "cohere.embed-english-v3", "cohere.embed-multilingual-v3"],
            },
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing models.json: {e}")
        raise


@lru_cache
def load_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load token pricing from models.json.

    Returns:
        Dict mapping provider -> model -> {prompt: X, completion: Y}
    """
    config = load_config()
    return config.get("pricing", {})


def get_llm_client(client_type: LLMService, **kwargs) -> "SimpleLLMClient":
    """
    Gets a chat client that satisfies SimpleLLMClient interface.

    Args:
        client_type: "openai", "ollama", "bedrock", or "groq"
        **kwargs: Additional keyword arguments specific to the chat client type:
            - model: str (optional, defaults from models.json if not provided)
    """
    client_type = client_type.lower()

    # Use config default model if not provided
    if "model" not in kwargs:
        config = load_config()
        default_models = config.get("chat_models", {}).get(client_type, [])
        if default_models:
            # Take the first model from the list as the default
            kwargs["model"] = default_models[0] if isinstance(default_models, list) else default_models

    if client_type == "openai":
        return OpenAIClient(**kwargs)
    if client_type == "ollama":
        return OllamaClient(**kwargs)
    if client_type == "bedrock":
        return BedrockClient(**kwargs)
    if client_type == "groq":
        return GroqClient(**kwargs)
    raise ValueError(f"Unknown chat client type: {client_type}")


class SimpleLLMClient(ABC):
    """Abstract base class for SimpleLLMClient, an API for LLM with async interface"""

    @abstractmethod
    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get a chat completion from the model.

        Note:
            - All implementations should handle errors gracefully by returning
              the standardized error format rather than raising exceptions
            - Tool/function calling support varies by provider
            - Token counting methods may vary between providers

        Args:
            messages: List of message dictionaries representing the conversation history.
                Each message dict must contain:
                - 'role': str - One of 'system', 'user', 'assistant', or 'tool'
                - 'content': str | None - The message content/text (None allowed for assistant with tool_calls)
                - 'name': str (optional) - Name of the message sender

                Message types:

                1. System messages (role='system'):
                   - 'role': 'system' (required)
                   - 'content': str (required) - System instructions or context
                   - 'name': str (optional) - Name identifier

                2. User messages (role='user'):
                   - 'role': 'user' (required)
                   - 'content': str (required) - User's message text
                   - 'name': str (optional) - Name identifier

                3. Assistant messages (role='assistant'):
                   - 'role': 'assistant' (required)
                   - 'content': str | None (optional) - Assistant's response text
                     Can be None if only tool_calls are present
                   - 'tool_calls': list[dict] (optional) - List of tool calls when assistant wants to use tools
                     Each tool call dict contains:
                     - 'id': str - Unique identifier for this tool call
                     - 'type': str - Usually 'function'
                     - 'function': dict - Function call details:
                       - 'name': str - Name of the function to call
                       - 'arguments': str - JSON string of function arguments
                   - 'name': str (optional) - Name identifier

                4. Tool messages (role='tool'):
                   - 'role': 'tool' (required)
                   - 'content': str (required) - Result of the tool execution
                   - 'tool_call_id': str (required) - ID of the tool call this result corresponds to
                   - 'status': str (optional) - Status of tool execution: 'success' or 'error'
                   - 'name': str (optional) - Name identifier

                Example:
                [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'What is the weather in Paris?'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'call_123',
                                'type': 'function',
                                'function': {
                                    'name': 'get_weather',
                                    'arguments': '{"location": "Paris"}'
                                }
                            }
                        ]
                    },
                    {
                        'role': 'tool',
                        'content': 'Sunny, 22°C',
                        'tool_call_id': 'call_123',
                        'status': 'success'
                    },
                    {'role': 'assistant', 'content': 'The weather in Paris is sunny and 22°C.'}
                ]

            tools: Optional list of tool/function definitions for function calling.
                Each tool dict must contain:
                - 'type': str - Tool type (typically 'function')
                - 'function': dict - Function specification sub-dictionary containing:
                  - 'name': str (required) - Function name, must be unique
                  - 'description': str (required) - Function description explaining what it does
                  - 'parameters': dict (required) - JSON Schema object defining the function parameters
                    The parameters dict must follow JSON Schema format with:
                    - 'type': str - Usually 'object'
                    - 'properties': dict - Dictionary of parameter definitions
                      Each property should have 'type' (e.g., 'string', 'number', 'boolean')
                    - 'required': list[str] (optional) - List of required parameter names

                Example:
                [{
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'description': 'Get current weather for a location',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'location': {
                                    'type': 'string',
                                    'description': 'City name or location'
                                },
                                'unit': {
                                    'type': 'string',
                                    'enum': ['celsius', 'fahrenheit'],
                                    'description': 'Temperature unit'
                                }
                            },
                            'required': ['location']
                        }
                    }
                }]

            max_tokens: Maximum number of tokens to generate in the completion.
                If None, uses the model's default limit. Different models have
                different default and maximum token limits.

            temperature: Controls randomness in generation (0.0 to 1.0).
                - 0.0: Deterministic, always picks most likely token
                - 1.0: Maximum randomness
                - Values between 0.1-0.7 are typically good for most use cases

        Returns:
            Dict[str, Any]: Standardized response dictionary containing:
            {
                'text': str,
                'metadata': {
                    'usage': {
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'finish_reason': str
                },
                'tool_calls': [
                    {
                        'function': {
                            'name': str,
                            'arguments': str,
                            'tool_call_id': str
                        }
                    }
                ] | None
            }

            On error, returns:
            {
                'text': 'Error: <error_message>',
                'metadata': {
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'elapsed_seconds': float
                    },
                    'model': str,
                    'error': str
                }
            }

        """
        pass

    @abstractmethod
    async def embed(self, input: str) -> List[float]:
        """Generate a text embedding vector for the given input string."""
        pass

    @abstractmethod
    def get_token_cost(self) -> float:
        """Get the cost per 1K tokens for the model in AUD."""
        pass

    def _build_error_response(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Build standardized error response structure.

        Args:
            error: The exception that occurred
            start_time: Request start time for elapsed calculation

        Returns:
            Dict with error information and metadata
        """
        return {
            "text": f"Error: {str(error)}",
            "metadata": {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_seconds": time.time() - start_time,
                },
                "model": self.model,
                "error": str(error),
            },
        }

    def _build_usage_metadata(
        self, prompt_tokens: int, completion_tokens: int, elapsed_seconds: float
    ) -> Dict[str, Any]:
        """Build standardized usage metadata structure.

        Returns:
            Dict with token counts and elapsed time
        """
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_seconds": elapsed_seconds,
        }

    def _build_success_response(
        self,
        text: str,
        usage: Dict[str, Any],
        finish_reason: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build standardized success response structure.

        Args:
            text: Response text content
            usage: Usage metadata from _build_usage_metadata()
            finish_reason: Why generation stopped
            tool_calls: Optional list of tool calls

        Returns:
            Dict with response data and metadata
        """
        result = {
            "text": text,
            "metadata": {
                "usage": usage,
                "model": self.model,
                "finish_reason": finish_reason,
            },
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def _format_tool_call_output(
        self, name: str, arguments: str, tool_call_id: str
    ) -> Dict[str, Any]:
        """Format tool call into standardized output structure.

        Args:
            name: Function/tool name
            arguments: JSON string of arguments
            tool_call_id: Unique identifier for this tool call

        Returns:
            Dict with function call details
        """
        return {
            "function": {
                "name": name,
                "arguments": arguments,
                "tool_call_id": tool_call_id,
            }
        }

    async def connect(self):
        """Initialize async resources. Override in subclasses as needed."""
        pass

    async def close(self):
        """Clean up async resources. Override in subclasses as needed."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def parse_response_as_json_list(response: Dict[str, Any] | str) -> Optional[Dict[str, Any] | List[Any]]:
    """Parse JSON from text response, extracting from markdown or .transactions if needed.

    Args:
        response: Response dict with 'text' key or raw string

    Returns:
        Parsed JSON object (dict or list) or None if parsing fails
    """
    if isinstance(response, dict):
        response_text = response.get("text", "")
    elif isinstance(response, str):
        response_text = response
    else:
        return None

    if not response_text:
        return None

    def try_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    parsed = try_parse(response_text)
    if parsed:
        return parsed

    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",
        r"```(?:json)?\s*({[\s\S]*})\s*```",
        r"\{[\s\S]*\}",
        r"({[\s\S]*})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            parsed = try_parse(match)
            if parsed:
                return parsed

    return None


class OllamaClient(SimpleLLMClient):
    def __init__(self, model: str = None):
        """Initialize Ollama chat client.

        Args:
            model: Name of the Ollama model to use (default from config)

        Raises:
            RuntimeError: If Ollama is not running or the model is not available
        """
        self.model = model
        self.client = None

    def _normalize_tool_call(self, tool_call: Any) -> Optional[Dict[str, str]]:
        """Normalize tool call from dict or object format to standard dict.

        Returns None if tool_call format is unrecognized.
        """
        function_name = None
        function_args = None
        tool_call_id = None

        # Handle dict format
        if isinstance(tool_call, dict) and "function" in tool_call:
            func = tool_call["function"]
            function_name = func.get("name", "")
            function_args = func.get("arguments", {})
            tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
        # Handle object format
        elif hasattr(tool_call, "function"):
            function = tool_call.function
            function_name = getattr(function, "name", "")
            function_args = getattr(function, "arguments", {})
            tool_call_id = getattr(tool_call, "id", f"call_{uuid.uuid4().hex[:8]}")
        else:
            logger.warning(f"Unknown tool_call format: {type(tool_call)}")
            return None

        # Convert args to JSON string if needed
        if isinstance(function_args, dict):
            function_args = json.dumps(function_args)
        elif not isinstance(function_args, str):
            function_args = str(function_args)

        return self._format_tool_call_output(function_name, function_args, tool_call_id)

    def _normalize_ollama_response(self, response: Any) -> Dict[str, Any]:
        """Normalize Ollama response from dict or object format.

        Ollama returns dict when streaming=False, object when streaming=True (default).
        Maps Ollama's 'done_reason' to standard 'finish_reason'.
        """
        if isinstance(response, dict):
            return {
                "message": response.get("message", {}),
                "finish_reason": response.get("done_reason", "stop")
            }
        else:
            message_obj = getattr(response, "message", None)
            finish_reason = getattr(response, "done_reason", "stop")

            if message_obj is not None and hasattr(message_obj, "model_dump"):
                message_dict = message_obj.model_dump()
            elif message_obj is not None:
                message_dict = {
                    "role": getattr(message_obj, "role", "assistant"),
                    "content": getattr(message_obj, "content", ""),
                }
            else:
                message_dict = {}

            return {"message": message_dict, "finish_reason": finish_reason}

    def _ensure_dict_arguments(self, arguments: Any) -> dict:
        """Convert function arguments to dict format (required by Ollama API).

        Ollama requires tool arguments as dict, not JSON string.
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                return {}
        elif isinstance(arguments, dict):
            return arguments
        else:
            return {}

    async def connect(self):
        if self.client:
            return

        logger.info(f"Initializing 'ollama:{self.model}'")
        self.client = ollama.AsyncClient()
        try:
            await self.client.list()
        except Exception as e:
            raise RuntimeError(
                "Ollama is not running or not installed. "
                "Please start the Ollama service and try again."
            ) from e

        try:
            ollama.show(self.model)
        except Exception as e:
            raise RuntimeError(
                f"Model '{self.model}' is not available. "
                f"Please ensure the model is pulled and available. Error: {str(e)}"
            )

    def _transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform standardized message format to Ollama's expected format.

        Ollama expects tool_calls arguments as dicts, not JSON strings.
        """
        transformed = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                new_msg = dict(msg)
                new_msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    new_tc = dict(tc)
                    if "function" in new_tc:
                        new_func = dict(new_tc["function"])
                        args = new_func.get("arguments", {})
                        new_func["arguments"] = self._ensure_dict_arguments(args)
                        new_tc["function"] = new_func
                    new_msg["tool_calls"].append(new_tc)
                transformed.append(new_msg)
            else:
                transformed.append(msg)
        return transformed

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Ollama implementation of get_completion with tool support."""
        await self.connect()

        start_time = time.time()

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            transformed_messages = self._transform_messages(messages)

            # Validate that all tool_calls have dict arguments before sending to Ollama
            for msg in transformed_messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        if "function" in tc:
                            args = tc["function"].get("arguments")
                            if isinstance(args, str):
                                logger.error(
                                    f"Tool call still has string arguments after transformation: "
                                    f"tool={tc['function'].get('name')}, args={args[:100]}"
                                )
                                # Force convert to dict as fallback
                                try:
                                    tc["function"]["arguments"] = json.loads(args) if args else {}
                                except json.JSONDecodeError:
                                    tc["function"]["arguments"] = {}

            chat_kwargs = {
                "model": self.model,
                "messages": transformed_messages,
                "options": options,
            }
            if tools is not None:
                chat_kwargs["tools"] = tools

            response = await self.client.chat(**chat_kwargs)
            elapsed_seconds = time.time() - start_time

            normalized = self._normalize_ollama_response(response)
            message_dict = normalized["message"]
            finish_reason = normalized["finish_reason"]

            response_text = (message_dict.get("content") or "") if isinstance(message_dict, dict) else ""
            raw_tool_calls = message_dict.get("tool_calls") if isinstance(message_dict, dict) else None
            
            completion_tokens = len(response_text.split()) if response_text else 0
            prompt_tokens = sum(len((m.get("content") or "").split()) for m in messages)
            total_tokens = prompt_tokens + completion_tokens

            tool_calls = None
            if raw_tool_calls:
                tool_calls = []
                for tool_call in raw_tool_calls:
                    normalized = self._normalize_tool_call(tool_call)
                    if normalized:
                        tool_calls.append(normalized)

            usage = self._build_usage_metadata(prompt_tokens, completion_tokens, elapsed_seconds)
            return self._build_success_response(response_text, usage, finish_reason, tool_calls)
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return self._build_error_response(e, start_time)

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using Ollama's embedding capabilities."""
        await self.connect()

        try:
            response = await self.client.embeddings(model=self.model, prompt=input)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error calling Ollama embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(self) -> float:
        """Returns 0.0 AUD since Ollama runs locally with no API costs."""
        return 0.0


class OpenAIClient(SimpleLLMClient):
    def __init__(
        self,
        model: str = None,
    ):
        """Initialize OpenAI chat client.

        Args:
            model: Name of the OpenAI model to use (default from config)

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
            RuntimeError: If the API key is invalid or the model is not available
        """
        self.model = model
        self.client = None
        self._closed = True

    async def connect(self):
        if self.client and not self._closed:
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )

        logger.info(f"Initializing 'openai:{self.model}'")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._closed = False

        try:
            await self.client.models.retrieve(self.model)
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "Invalid OpenAI API key. Please check your API key and try again."
            ) from e
        except openai.NotFoundError as e:
            raise RuntimeError(
                f"Model '{self.model}' not found or you don't have access to it. "
                f"Please check the model name and your API permissions."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI API: {str(e)}") from e

    async def close(self):
        """Close the OpenAI client and release resources."""
        if self.client is not None and not self._closed:
            await self.client.close()
            self.client = None
            self._closed = True

    def _handle_incomplete_tool_sequence(self, msg: Dict, i: int, messages: List[Dict]) -> Optional[Dict]:
        """Handle assistant message with tool_calls that has no following tool messages.

        This occurs when loading conversation history with incomplete tool sequences.
        OpenAI requires: assistant (with tool_calls) → tool → assistant

        Returns cleaned message without tool_calls, or None to skip.
        """
        # Check if next message is a tool message
        has_following_tool = (
            i + 1 < len(messages) and messages[i + 1].get("role") == "tool"
        )

        if not has_following_tool:
            # Incomplete sequence from history - strip tool_calls
            clean_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not clean_msg.get("content"):
                clean_msg["content"] = ""
            logger.debug("Stripped tool_calls from assistant message (incomplete sequence)")
            return clean_msg

        return msg

    def _should_skip_orphaned_tool_message(self, in_active_sequence: bool) -> bool:
        """Check if tool message should be skipped because it's orphaned.

        Tool messages are only valid within an active tool sequence started by
        an assistant message with tool_calls.
        """
        if not in_active_sequence:
            logger.debug("Skipping orphaned tool message (no active tool sequence)")
            return True
        return False

    def _transform_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform intermediate message format to OpenAI API format.

        OpenAI requires strict message sequencing for tool calls:
        - assistant (with tool_calls) → tool (with tool_call_id) → assistant

        When building messages from conversation history, we filter out incomplete
        tool sequences to avoid 400 validation errors.
        """
        formatted_messages = []
        # Track if we're in an active tool sequence (assistant with tool_calls → tool messages)
        # This is needed because OpenAI validates strict message sequencing
        in_active_tool_sequence = False

        for i, msg in enumerate(messages):
            role = msg["role"]

            # System messages always end any previous sequence
            if role == "system":
                in_active_tool_sequence = False
                formatted_messages.append(msg)
                continue

            # Tool messages are only valid within an active sequence
            if role == "tool":
                if not self._should_skip_orphaned_tool_message(in_active_tool_sequence):
                    formatted_messages.append({
                        "role": "tool",
                        "content": msg.get("content", ""),
                        "tool_call_id": msg.get("tool_call_id", ""),
                    })
            # Assistant messages
            elif role == "assistant":
                # Handle incomplete tool sequences from history
                if "tool_calls" in msg:
                    cleaned_msg = self._handle_incomplete_tool_sequence(msg, i, messages)
                    if cleaned_msg:
                        # Check if we kept tool_calls (complete sequence) or stripped them (incomplete)
                        if "tool_calls" in cleaned_msg:
                            in_active_tool_sequence = True
                            formatted_messages.append(cleaned_msg)
                        else:
                            in_active_tool_sequence = False
                            formatted_messages.append(cleaned_msg)
                else:
                    in_active_tool_sequence = False
                    formatted_messages.append(msg)
            elif role == "user":
                in_active_tool_sequence = False
                formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)

        return formatted_messages

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """OpenAI implementation of get_completion with full tool support."""
        await self.connect()

        start_time = time.time()

        try:
            formatted_messages = self._transform_messages(messages)
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed_seconds = time.time() - start_time

            text = completion.choices[0].message.content if completion.choices else ""

            if hasattr(completion, "usage") and completion.usage:
                usage = self._build_usage_metadata(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    elapsed_seconds,
                )
            else:
                usage = self._build_usage_metadata(0, 0, elapsed_seconds)

            # Extract tool calls if present
            tool_calls = None
            if completion.choices and completion.choices[0].message.tool_calls:
                tool_calls = []
                for tool_call in completion.choices[0].message.tool_calls:
                    tool_calls.append(
                        self._format_tool_call_output(
                            tool_call.function.name,
                            tool_call.function.arguments,
                            tool_call.id,
                        )
                    )

            finish_reason = (
                completion.choices[0].finish_reason
                if completion.choices and completion.choices[0].finish_reason
                else "stop"
            )
            return self._build_success_response(text, usage, finish_reason, tool_calls)
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return self._build_error_response(e, start_time)

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using OpenAI's embedding model."""
        await self.connect()

        try:
            response = await self.client.embeddings.create(
                model=self.model, input=input
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error calling OpenAI embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """Calculate token cost based on OpenAI pricing."""
        pricing = load_pricing().get("openai", {})
        model_pricing = pricing.get(self.model)

        if model_pricing:
            prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
            completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
            return prompt_cost + completion_cost

        logger.warning(f"No pricing data for model: {self.model}")
        return None


class GroqClient(OpenAIClient):
    """Groq chat client that inherits from OpenAI client (Groq uses OpenAI-compatible API).

    Groq provides fast inference for open-source models using their custom LPU hardware.
    Inherits all functionality from OpenAIClient including:
    - Tool/function calling support
    - Standard message formatting
    - Error handling with _build_error_response()
    - Usage tracking with _build_usage_metadata()
    - Response formatting with _build_success_response()
    """

    async def connect(self):
        """Initialize connection to Groq API.

        Raises:
            ValueError: If GROQ_API_KEY environment variable is not set
        """
        if self.client and not self._closed:
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set GROQ_API_KEY in your .env file or environment variables."
            )

        logger.info(f"Initializing 'groq:{self.model}'")
        self.client = groq.AsyncGroq(api_key=api_key)
        self._closed = False

    async def embed(self, input: str) -> List[float]:
        """Groq does not currently support embeddings."""
        raise NotImplementedError(
            "Groq does not currently support text embeddings. "
            "Please use OpenAI or another provider for embedding generation."
        )

    def get_token_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """Calculate token cost based on Groq pricing."""
        pricing = load_pricing().get("groq", {})
        model_pricing = pricing.get(self.model)

        if model_pricing:
            prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
            completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
            return prompt_cost + completion_cost

        logger.warning(f"No pricing data for Groq model: {self.model}")
        return None


def _read_aws_profiles_from_file(file_path: str) -> Set[str]:
    """Read AWS profile names from config file.

    Args:
        file_path: Path to credentials or config file

    Returns:
        Set of profile names found in file
    """
    if not os.path.exists(file_path):
        return set()

    config = configparser.ConfigParser()
    config.read(file_path)

    profiles = set()
    for section in config.sections():
        # Strip 'profile ' prefix from config file sections
        if section.startswith("profile "):
            profiles.add(section.replace("profile ", "", 1))
        else:
            profiles.add(section)
    return profiles


def _discover_aws_profiles() -> Set[str]:
    """Discover all available AWS profiles from credentials and config files.

    Returns:
        Set of all profile names
    """
    home_dir = Path.home()
    credentials_path = home_dir / ".aws" / "credentials"
    config_path = home_dir / ".aws" / "config"

    profiles = set()
    profiles.update(_read_aws_profiles_from_file(str(credentials_path)))
    profiles.update(_read_aws_profiles_from_file(str(config_path)))

    return profiles


def _check_sso_expiration(session: Any, profile_name: str) -> None:
    """Check if SSO session is expired and provide helpful error message.

    Args:
        session: boto3 Session object
        profile_name: Name of the AWS profile

    Raises:
        ValueError: If SSO session is expired with instructions
    """
    # Check for expired SSO
    home_dir = Path.home()
    sso_cache_dir = home_dir / ".aws" / "sso" / "cache"

    if sso_cache_dir.exists():
        for cache_file in sso_cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    cache_data = json.load(f)
                    if "expiresAt" in cache_data:
                        expires_at = datetime.fromisoformat(
                            cache_data["expiresAt"].replace("Z", "+00:00")
                        )
                        if expires_at < datetime.now(timezone.utc):
                            raise ValueError(
                                f"AWS SSO session expired for profile '{profile_name}'. "
                                f"Run: aws sso login --profile {profile_name}"
                            )
            except (json.JSONDecodeError, ValueError, KeyError):
                continue


def _handle_aws_credential_error(error: Exception, profile_name: str) -> str:
    """Generate helpful error message for AWS credential errors.

    Args:
        error: The credential error that occurred
        profile_name: Name of the AWS profile

    Returns:
        User-friendly error message with troubleshooting steps
    """
    error_msg = str(error).lower()

    if "could not be found" in error_msg or "profile" in error_msg:
        return (
            f"AWS profile '{profile_name}' not found. "
            f"Check ~/.aws/credentials or ~/.aws/config"
        )
    elif "sso" in error_msg or "token" in error_msg:
        return (
            f"AWS SSO error for profile '{profile_name}'. "
            f"Run: aws sso login --profile {profile_name}"
        )
    elif "credentials" in error_msg or "access" in error_msg:
        return (
            f"AWS credentials error for profile '{profile_name}'. "
            f"Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
    else:
        return f"AWS configuration error: {error}"


@lru_cache(maxsize=None)
def get_aws_config(is_raise_exception: bool = True) -> Dict[str, Any]:
    """
    Returns AWS configuration for boto3 client initialization.

    This function searches for AWS profiles and saved credentials to build a
    configuration dictionary that can be used to initialize boto3 clients and
    sessions. It validates the discovered credentials to ensure they are properly
    configured and not expired.

    Credential Discovery Process:
    1. Looks for AWS_PROFILE environment variable to determine profile name
    2. Searches for saved credentials in ~/.aws/credentials and ~/.aws/config
    3. Validates the profile exists if AWS_PROFILE is set
    4. Falls back to default credential chain if profile not found
    5. Creates a boto3 session using the discovered profile (or default)
    6. Validates credentials contain required access_key and secret_key
    7. Tests credential validity with an STS GetCallerIdentity call
    8. Checks for SSO token expiration on SSO profiles

    Environment Variables:
        AWS_PROFILE (str, optional): Name of the AWS profile to use from
                                   ~/.aws/credentials. If not set, uses the
                                   default profile.
        AWS_REGION (str, optional): AWS region to use for AWS services

    Returns:
        dict: AWS configuration dictionary for boto3 client initialization:
            - profile_name (str, optional): The AWS profile name to pass to
                                          boto3.client() or boto3.Session()
            - region_name (str): AWS region name for service clients

    Note:
        This function is cached to avoid repeated credential discovery and
        validation. The returned configuration can be unpacked directly into
        boto3 client constructors. All validation errors are logged but do not
        raise exceptions - returns gracefully with empty config on failure.

    Examples:
        >>> aws_config = get_aws_config()
        >>> s3_client = boto3.client('s3', **aws_config)
        >>>
        >>> # Or with session
        >>> session = boto3.Session(**aws_config)
        >>> dynamodb = session.client('dynamodb')
    """
    aws_config = {}

    # Discover available profiles
    available_profiles = _discover_aws_profiles()

    # Validate AWS_PROFILE exists if specified
    profile_name = os.getenv("AWS_PROFILE")
    profile_not_found = False
    if profile_name:
        if profile_name in available_profiles:
            aws_config["profile_name"] = profile_name
        else:
            logger.info(f"AWS profile '{profile_name}' not found, using default credential chain")
            profile_not_found = True

    region = os.getenv("AWS_REGION")
    if region:
        aws_config["region_name"] = region

    # Remove AWS_PROFILE from env if profile not found to allow fallback
    if profile_not_found:
        os.environ.pop("AWS_PROFILE", None)

    try:
        session = boto3.Session(**aws_config)
        credentials = session.get_credentials()

        if not credentials:
            if is_raise_exception:
                if available_profiles:
                    raise ValueError(
                        f"No AWS credentials found.\n"
                        f"Available profiles: {', '.join(available_profiles)}\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
                else:
                    raise ValueError(
                        f"No AWS credentials found.\n"
                        f"To configure: aws configure\n"
                        f"Or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables"
                    )
            return aws_config

        if not credentials.access_key or not credentials.secret_key:
            if is_raise_exception:
                raise ValueError("Incomplete AWS credentials (missing access key or secret key)")
            logger.warning("Incomplete AWS credentials")
            return aws_config

        # Validate credentials work
        sts = session.client("sts")
        sts.get_caller_identity()

        # Check for SSO expiry with helpful error message
        if profile_name and profile_name in aws_config.get("profile_name", ""):
            try:
                _check_sso_expiration(session, profile_name)
            except ValueError as e:
                if is_raise_exception:
                    raise
                logger.warning(str(e))
                return aws_config

        return aws_config

    except ClientError as e:
        if is_raise_exception:
            raise
        error_code = e.response["Error"]["Code"]

        if error_code == "ExpiredToken":
            profile_to_check = aws_config.get("profile_name", profile_name)
            if profile_to_check:
                logger.warning(f"AWS SSO session expired for profile '{profile_to_check}'. Run: aws sso login --profile {profile_to_check}")
            else:
                logger.warning("AWS credentials have expired")
        elif error_code == "InvalidClientTokenId":
            logger.warning("AWS credentials are invalid. Please reconfigure: aws configure")
        else:
            logger.warning(f"AWS API error: {error_code}")
    except Exception as e:
        if is_raise_exception:
            raise
        profile_to_check = aws_config.get("profile_name", profile_name)
        error_msg = _handle_aws_credential_error(e, profile_to_check or "default")
        logger.error(error_msg)

    return aws_config


class BedrockClient(SimpleLLMClient):
    def __init__(
        self,
        model: str = None,
    ):
        """
        Initialize Bedrock chat client.

        This implementation exclusively uses the Bedrock Converse API to enable
        tool/function calling support. As a result, only Claude models are supported
        since they are the primary models that work well with the Converse API for
        tool usage. Other Bedrock models may not support tools through this API.

        Args:
            model: Claude model ID for Bedrock (default from config).
        """
        self.model = model
        self.client = None
        self._session = None
        self._closed = True

    async def connect(self):
        """Initialize the async client session and client."""
        if self.client is not None and not self._closed:
            return

        logger.info(f"Initializing 'bedrock:{self.model}'")
        aws_config = get_aws_config()
        self._session = aioboto3.Session(**aws_config)
        self.client = await self._session.client("bedrock-runtime").__aenter__()
        self._closed = False

    async def close(self):
        """Close the client and properly clean up aiohttp sessions."""
        if self.client is not None and not self._closed:
            await self.client.__aexit__(None, None, None)
            self.client = None
            self._closed = True

            # Give asyncio a moment to clean up pending tasks and connections
            # This ensures aiohttp connectors are properly closed
            await asyncio.sleep(AIOBOTO3_CLEANUP_DELAY_SECONDS)

    def _build_result_from_response(
        self, response: Any, start_time: float
    ) -> Dict[str, Any]:
        """Build result from Bedrock Converse API response.

        Bedrock Converse API returns responses in format:
        {
          "output": {
            "message": {
              "content": [
                {"text": "..."} or
                {"toolUse": {"toolUseId": "...", "name": "...", "input": {...}}}
              ]
            }
          },
          "usage": {"inputTokens": X, "outputTokens": Y, ...},
          "stopReason": "end_turn" | "tool_use" | ...
        }

        This method extracts text, tool calls, and usage data into our
        standardized response format.

        Args:
            response: Bedrock Converse API response (dict) or error string
            start_time: Request start time for elapsed calculation

        Returns:
            Standardized response dict with text, metadata, and optional tool_calls
        """
        response_text_blocks = []
        tool_calls = []

        if isinstance(response, str):
            response_text_blocks.append(response)
            usage = {}
            stop_reason = "stop"
        else:
            output = response.get("output", {})
            if isinstance(output, dict) and "message" in output:
                message = output["message"]
                for content in message.get("content", []):
                    if "text" in content:
                        response_text_blocks.append(content["text"])
                    elif "toolUse" in content:
                        tool_use = content["toolUse"]
                        tool_call_id = tool_use.get("toolUseId", "")
                        tool_calls.append(
                            self._format_tool_call_output(
                                tool_use["name"],
                                json.dumps(tool_use.get("input", {})),
                                tool_call_id,
                            )
                        )
            usage_dict = response.get("usage", {})
            stop_reason = response.get("stopReason", "unknown")

        text = "\n".join(response_text_blocks).strip()
        usage = self._build_usage_metadata(
            usage_dict.get("inputTokens", 0),
            usage_dict.get("outputTokens", 0),
            time.time() - start_time,
        )
        return self._build_success_response(text, usage, stop_reason, tool_calls if tool_calls else None)

    def _batch_consecutive_tool_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch consecutive tool messages into single user messages.

        Bedrock Converse API requires all tool results following an assistant
        message with tool calls to be in a single user message with multiple
        toolResult blocks in the content array.

        :param messages: List of messages in intermediate format
        :return: Messages with batched tool results
        """
        batched = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # If this is a tool message, batch it with any following tool messages
            if msg.get("role") == "tool":
                tool_batch = []

                # Collect all consecutive tool messages
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_batch.append(messages[i])
                    i += 1

                # Convert to Bedrock format: single user message with multiple toolResult blocks
                if len(tool_batch) > 1:
                    logger.debug(
                        f"Batching {len(tool_batch)} tool results into single user message"
                    )
                    combined_content = []
                    for tool_msg in tool_batch:
                        combined_content.append({
                            "toolResult": {
                                "toolUseId": tool_msg.get("tool_call_id", ""),
                                "content": [{"text": str(tool_msg.get("content", ""))}],
                                "status": tool_msg.get("status", "success"),
                            }
                        })

                    batched.append({
                        "role": "user",
                        "content": combined_content
                    })
                elif len(tool_batch) == 1:
                    # Single tool result: keep as-is, will be formatted below
                    batched.append(tool_batch[0])
            else:
                batched.append(msg)
                i += 1

        return batched

    def _transform_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Transform intermediate message format to Bedrock Converse API format.

        Batches consecutive tool messages before formatting to ensure Bedrock
        Converse API requirements are met (all tool results in single user message).

        Returns partially filled request_kwargs with messages, system, and toolConfig.
        """
        # Batch consecutive tool messages first
        messages = self._batch_consecutive_tool_messages(messages)

        system_parts = []
        formatted_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            elif role == "assistant" and "tool_calls" in msg:
                assistant_content = []
                if content:
                    assistant_content.append({"text": content})
                for tool_call in msg.get("tool_calls", []):
                    # Bedrock stores tool_call_id in function.tool_call_id (line 1206), not top-level 'id'
                    tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("tool_call_id", "")
                    if tool_call_id:
                        assistant_content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_call_id,
                                    "name": tool_call["function"]["name"],
                                    "input": json.loads(
                                        tool_call["function"]["arguments"]
                                    )
                                    if isinstance(
                                        tool_call["function"]["arguments"], str
                                    )
                                    else tool_call["function"]["arguments"],
                                }
                            }
                        )
                    else:
                        logger.warning(f"Tool call missing tool_call_id, skipping: {tool_call.get('function', {}).get('name', 'unknown')}")
                if assistant_content:
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content,
                        }
                    )
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                tool_content = (
                    content.rstrip() if isinstance(content, str) else str(content)
                )
                tool_status = msg.get("status", "success")
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": tool_call_id,
                                    "content": [{"text": tool_content}],
                                    "status": tool_status,
                                }
                            }
                        ],
                    }
                )
            elif (
                role == "user"
                and isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and "toolResult" in content[0]
            ):
                formatted_messages.append(msg)
            elif role == "assistant" and isinstance(content, list):
                formatted_messages.append(msg)
            else:
                role = "user" if role == "user" else "assistant"
                content = content.rstrip() if isinstance(content, str) else content
                formatted_messages.append(
                    {"role": role, "content": [{"text": content}]}
                )

        formatted_tools = None
        if tools:
            formatted_tools = [
                {
                    "toolSpec": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "inputSchema": {"json": tool["function"].get("parameters", {})},
                    }
                }
                for tool in tools
            ]

        system_blocks = [{"text": "\n\n".join(system_parts)}] if system_parts else []

        request_kwargs = {
            "messages": formatted_messages,
            "system": system_blocks,
        }

        if tools:
            request_kwargs["toolConfig"] = {"tools": formatted_tools}

        return request_kwargs

    async def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Get completion from Bedrock Converse API."""
        await self.connect()
        start_time = time.time()

        try:
            request_kwargs = self._transform_messages(messages, tools)
            request_kwargs.update(
                {
                    "modelId": self.model,
                    "inferenceConfig": {
                        "temperature": temperature,
                        "maxTokens": max_tokens or 1024,
                    },
                }
            )

            response = await self.client.converse(**request_kwargs)
            return self._build_result_from_response(response, start_time)

        except Exception as e:
            logger.error(f"Error in Bedrock get_completion: {e}")
            return self._build_error_response(e, start_time)

    async def embed(self, input: str) -> List[float]:
        """Generate text embeddings using Bedrock's embedding model."""
        try:
            await self.connect()

            response = await self.client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": input}),
            )

            raw_body = await response["body"].read()
            body = json.loads(raw_body.decode("utf-8"))
            return body["embedding"]

        except Exception as e:
            logger.error(f"Error calling Bedrock embed: {e}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def get_token_cost(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """Calculate token cost based on Bedrock pricing."""
        pricing = load_pricing().get("bedrock", {})

        # Try exact model match first
        model_pricing = pricing.get(self.model)

        # Fall back to simplified model name matching
        if not model_pricing:
            for model_key in pricing.keys():
                if model_key in self.model:
                    model_pricing = pricing[model_key]
                    break

        if model_pricing:
            prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
            completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
            return prompt_cost + completion_cost

        logger.warning(f"No pricing data for Bedrock model: {self.model}")
        return None
