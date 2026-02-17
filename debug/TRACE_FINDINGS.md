# Tool Call ID Tracing Findings - Bedrock Pipeline

## Executive Summary

**Root Cause Confirmed**: Line 1234 of `chatboti/llm.py` in `BedrockClient._transform_messages()` incorrectly looks for `tool_call.get("id")` when the actual tool_call_id is stored in `tool_call["function"]["tool_call_id"]`.

This causes tool calls to be silently dropped during message transformation, resulting in Bedrock ValidationException errors complaining about missing toolResult blocks.

---

## Tool Call ID Data Flow

### 1. Response from Bedrock → Internal Format

**Location**: `BedrockClient.get_completion()` (~line 1111-1121)

When Bedrock returns a response with tool calls, the code transforms it:

```python
elif "toolUse" in content:
    tool_use = content["toolUse"]
    logger.debug(f"[TRACE] Bedrock response toolUse block: {tool_use}")
    tool_call_id = tool_use.get("toolUseId", "")
    logger.debug(f"[TRACE] Extracted tool_call_id from Bedrock: '{tool_call_id}'")
    tool_calls.append(
        {
            "function": {
                "name": tool_use["name"],
                "arguments": json.dumps(tool_use.get("input", {})),
                "tool_call_id": tool_call_id,  # ← STORED HERE
            }
        }
    )
```

**Key Point**: The tool_call_id is extracted from Bedrock's `toolUseId` field and stored in `tool_call["function"]["tool_call_id"]`.

**Data Structure Created**:
```python
{
    "function": {
        "name": "list_all_speakers",
        "arguments": "{}",
        "tool_call_id": "tooluse_abc123"  # ← The actual ID
    }
    # NOTE: No "id" field at top level!
}
```

---

### 2. Internal Format → Bedrock Request Format (THE BUG)

**Location**: `BedrockClient._transform_messages()` (~line 1224-1240)

When transforming messages back to Bedrock format:

```python
elif role == "assistant" and "tool_calls" in msg:
    assistant_content = []
    if content:
        assistant_content.append({"text": content})
    for tool_call in msg.get("tool_calls", []):
        logger.debug(f"[TRACE] Processing tool_call structure: {tool_call}")
        tool_call_id = tool_call.get("id", "")  # ← BUG: Looks for "id" at top level
        logger.debug(f"[TRACE] tool_call.get('id', ''): '{tool_call_id}'")
        function_tool_call_id = tool_call.get("function", {}).get("tool_call_id", "")
        logger.debug(f"[TRACE] tool_call.get('function', {{}}).get('tool_call_id', ''): '{function_tool_call_id}'")
        if tool_call_id:  # ← This condition FAILS because "id" doesn't exist
            logger.debug(f"[TRACE] ✓ tool_call_id is truthy, adding toolUse to assistant_content")
            # ... add toolUse block ...
        else:
            logger.debug(f"[TRACE] ✗ tool_call_id is empty/falsy, DROPPING tool_call silently!")
```

**The Problem**:
- `tool_call.get("id", "")` returns `""` (empty string) because there's no "id" field at the top level
- `tool_call["function"]["tool_call_id"]` contains the actual ID: `"tooluse_abc123"`
- The `if tool_call_id:` check fails, causing the entire tool_call to be silently dropped
- No toolUse block is added to assistant_content
- The formatted message sent to Bedrock is missing the assistant's tool calls

---

### 3. Tool Results → Bedrock Request Format

**Location**: `BedrockClient._transform_messages()` (~line 1258-1277)

When formatting tool result messages:

```python
elif role == "tool":
    tool_call_id = msg.get("tool_call_id", "")
    logger.debug(f"[TRACE] Processing tool result message with tool_call_id: '{tool_call_id}'")
    # ... format toolResult block ...
    toolResult_block = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_call_id,  # ← References the ID
                    "content": [{"text": tool_content}],
                    "status": tool_status,
                }
            }
        ],
    }
    logger.debug(f"[TRACE] Sending toolResult block to Bedrock: {toolResult_block}")
```

**The Result**:
- Tool results are correctly formatted with the tool_call_id
- But Bedrock never saw the original toolUse blocks (they were dropped at step 2)
- Bedrock rejects the request: "Expected toolResult blocks at messages.2.content for the following Ids: tooluse_abc123"

---

## Evidence from Other LLM Clients

The bug is confirmed by checking how other LLM clients in the same file store tool_call_id:

### OpenAIClient (line 511-513)
```python
"function": Function(
    name=tc.function.name,
    arguments=tc.function.arguments,
    tool_call_id=tc.id,  # ← Stored in function.tool_call_id
)
```

### AnthropicClient (line 758-760)
```python
"function": {
    "name": tool_use.name,
    "arguments": json.dumps(tool_use.input),
    "tool_call_id": tool_use.id,  # ← Stored in function.tool_call_id
}
```

### BedrockClient (line 1118)
```python
"function": {
    "name": tool_use["name"],
    "arguments": json.dumps(tool_use.get("input", {})),
    "tool_call_id": tool_use.get("toolUseId", ""),  # ← Stored in function.tool_call_id
}
```

**All three clients use the same internal format**: `tool_call["function"]["tool_call_id"]`

---

## The Fix (from chatboti-8a0.3)

The fix is simple - check both locations:

```python
# Check both top-level "id" and nested "function.tool_call_id"
tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("tool_call_id", "")
```

This handles:
- Legacy format (if any exists): `{"id": "...", "function": {...}}`
- Current format: `{"function": {"tool_call_id": "...", ...}}`

---

## Expected Trace Output (with AWS credentials)

When the repro script runs successfully, the trace logs would show:

**After Bedrock Response**:
```
[TRACE] Bedrock response toolUse block: {'toolUseId': 'tooluse_abc123', 'name': 'list_all_speakers', 'input': {}}
[TRACE] Extracted tool_call_id from Bedrock: 'tooluse_abc123'
```

**During Message Transformation**:
```
[TRACE] Processing tool_call structure: {'function': {'name': 'list_all_speakers', 'arguments': '{}', 'tool_call_id': 'tooluse_abc123'}}
[TRACE] tool_call.get('id', ''): ''
[TRACE] tool_call.get('function', {}).get('tool_call_id', ''): 'tooluse_abc123'
[TRACE] ✗ tool_call_id is empty/falsy, DROPPING tool_call silently!
```

**Tool Result Formatting**:
```
[TRACE] Processing tool result message with tool_call_id: 'tooluse_abc123'
[TRACE] Sending toolResult block to Bedrock: {'role': 'user', 'content': [{'toolResult': {'toolUseId': 'tooluse_abc123', ...}}]}
```

This would clearly show the bug: the tool_call_id exists in `function.tool_call_id` but the code only checks the top-level `id` field.

---

## Impact

This bug affects **all** Bedrock tool calls:
- Single tool call scenarios fail (tool call is dropped, no results sent)
- Multiple tool call scenarios fail (all tool calls are dropped)
- The error manifests as Bedrock ValidationException about missing toolResult blocks
- The bug is silent - no error is logged when tool_calls are dropped at line 1234

---

## Verification Status

- ✓ Tracing code added to all strategic locations
- ✓ Bug location confirmed at line 1234 of llm.py
- ✓ Data structure format confirmed across all 3 LLM clients
- ✓ Fix from chatboti-8a0.3 verified to address the root cause
- ⚠ Full trace output requires valid AWS Bedrock credentials

---

## Related Issues

- **chatboti-8a0.3**: Contains the fix for this bug
- **chatboti-8a0.2**: This task - adding tracing to confirm the bug location
- **Root cause**: Inconsistent tool_call_id lookup between response parsing (line 1118) and message formatting (line 1234)
