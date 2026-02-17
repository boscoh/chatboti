# Bedrock Multi-Tool-Call Debug Reproduction Script

## Overview

This repository now contains `debug_bedrock_repro.py`, a standalone reproduction script for debugging the Bedrock multi-tool-call validation error that occurs in the flaky test `test_multi_step_tool_chaining[bedrock]`.

## Problem Statement

The test `test_multi_step_tool_chaining[bedrock]` has a ~30% pass rate. When it fails, we get:

```
Error: An error occurred (ValidationException) when calling the Converse operation:
Expected toolResult blocks at messages.2.content for the following Ids: tooluse_S6hhcw36PqoKJAUBuu9hmw
```

**Hypothesis**: When Bedrock generates 2+ simultaneous tool calls, one tool result may appear missing due to message batching or formatting issues in the `BedrockClient._batch_consecutive_tool_messages()` or `_transform_messages()` methods.

## Why This Script Was Needed

Debugging was previously blocked because:
- pytest's module-scoped fixtures cache `BedrockClient` instances
- Debug logs in `BedrockClient.__init__()` and `connect()` never execute for cached instances
- Cannot see the actual message transformation happening before API calls

## What The Script Does

The script bypasses pytest entirely and:

1. **Creates a fresh BedrockClient** instance (no caching)
2. **Simulates the exact test scenario**:
   - User query: "Use list_all_speakers, then get_matching_speakers..."
   - Available tools: `list_all_speakers`, `get_matching_speakers`
   - Two tool calls expected from the LLM
3. **Logs comprehensive debug information**:
   - Initial messages sent to Bedrock
   - Raw Bedrock Converse API request structure (messages, tools, system)
   - Raw Bedrock Converse API response (including toolUse blocks with IDs)
   - Tool result formatting (toolResult blocks)
   - Transformed messages before the second API call
4. **Mock tool results** to avoid needing MCP server
5. **Catches and reports** the ValidationException if it occurs

## Usage

### Prerequisites

- Valid AWS credentials configured (AWS_PROFILE or ~/.aws/credentials)
- Access to Bedrock service in your AWS account
- `bedrock:InvokeModel` permission for `amazon.nova-pro-v1:0`

### Running the Script

```bash
# From the worktree directory
cd /Users/boscoh/p/chatboti-8a0.1

# Run with uv to ensure correct environment
uv run python debug_bedrock_repro.py
```

### Expected Output

The script will print detailed section headers showing:

```
================================================================================
  BEDROCK REPRO SCRIPT STARTED
================================================================================

================================================================================
  TOOLS DEFINED
================================================================================
  [0] list_all_speakers: Get list of all available speaker names.
  [1] get_matching_speakers: Find the top N most relevant speakers...

================================================================================
  INITIAL MESSAGES (Before First API Call)
================================================================================
...

================================================================================
  FIRST API CALL
================================================================================
...

================================================================================
  BEDROCK CONVERSE API REQUEST
================================================================================
Model ID: amazon.nova-pro-v1:0

Messages: 2
  Message [0] Role: user
    Content blocks: 1
      [0] text: You are a helpful assistant...

================================================================================
  BEDROCK CONVERSE API RESPONSE
================================================================================
Stop Reason: tool_use

Content blocks: 2
  Block [0]:
    toolUse:
      toolUseId: tooluse_abc123
      name: list_all_speakers
      input: {}

  Block [1]:
    toolUse:
      toolUseId: tooluse_xyz789
      name: get_matching_speakers
      input: {"query": "machine learning", "n": 3}

================================================================================
  MESSAGES BEFORE SECOND API CALL (With Tool Results)
================================================================================
[Message details showing all tool results...]

================================================================================
  BEDROCK CONVERSE API REQUEST
================================================================================
[Transformed message structure - THIS IS WHERE THE BUG MANIFESTS]

================================================================================
  ERROR OCCURRED (Issue Reproduced!)
================================================================================
```

If the bug occurs, you'll see which tool IDs Bedrock expects vs. which ones are actually present in the request.

## Key Files to Investigate

Based on the error, these are the critical code paths:

1. **`chatboti/llm.py:BedrockClient._batch_consecutive_tool_messages()`**
   - Lines 1145-1198
   - Batches consecutive tool messages into single user messages
   - May be dropping or incorrectly formatting tool results

2. **`chatboti/llm.py:BedrockClient._transform_messages()`**
   - Lines 1200-1313
   - Transforms messages to Bedrock Converse API format
   - Calls `_batch_consecutive_tool_messages()` first

3. **`chatboti/agent.py:InfoAgent._build_assistant_message()`**
   - Lines 235-258
   - Formats assistant message with tool_calls
   - Ensures tool_call IDs are properly set

4. **`chatboti/agent.py:InfoAgent.process_query()`**
   - Lines 370-455
   - Main tool execution loop
   - Appends tool results to message history

## Debug Strategy

With this script, you can now:

1. **Add print statements** or debug logging to `BedrockClient` methods
2. **See the exact message format** sent to Bedrock
3. **Compare tool IDs** in the assistant message vs. tool result messages
4. **Test fixes** by modifying `chatboti/llm.py` and re-running the script
5. **Verify the fix** works before running the full pytest suite

## Next Steps

1. Run the script with valid AWS credentials
2. Examine the logged message structures when the error occurs
3. Identify which tool result is "missing" according to Bedrock
4. Check if the tool_call_id is being:
   - Lost during message formatting
   - Duplicated (causing one to be skipped)
   - Incorrectly batched with another tool result
5. Fix the bug in `BedrockClient`
6. Re-run the script to verify the fix
7. Run the full test suite to confirm the flaky test is now stable

## Branch Information

- **Branch**: `task/bedrock-repro-script`
- **Remote**: `origin/task/bedrock-repro-script`
- **Commit**: Created standalone repro script bypassing pytest fixture caching

## Related Issues

- Beads task: `chatboti-8a0.1`
- Test failure log: `/tmp/bedrock_debug.log`
- Original failing test: `tests/test_tool_usage.py::test_multi_step_tool_chaining[bedrock]`
