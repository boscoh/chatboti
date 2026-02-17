# Debug Utilities

This directory contains debugging utilities and investigation artifacts.

## Bedrock Tool Call Bug Investigation

**Files:**
- `debug_bedrock_repro.py` - Standalone reproduction script for Bedrock tool call issues
- `DEBUG_BEDROCK_REPRO_README.md` - Usage instructions for the repro script
- `TRACE_FINDINGS.md` - Investigation findings and root cause analysis

**Bug Fixed:** 2026-02-17
The tool_call_id extraction bug in `chatboti/llm.py` line 1235 has been fixed.
These files are kept for future reference and debugging similar issues.

**Root Cause:**
Tool calls were being silently dropped because the code only checked `tool_call.get("id")`,
but Bedrock stores the ID in `tool_call["function"]["tool_call_id"]`.

**Fix:**
```python
tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("tool_call_id", "")
```

See git history and chatboti-8a0.x Beads tasks for full investigation details.
