#!/usr/bin/env python3
"""PreToolUse hook: block reads/edits to .env files containing credentials."""

import json
import sys

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})
file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

if file_path and (
    file_path == ".env" or file_path.endswith(("/.env",)) or "/.env." in file_path
):
    print(
        "Blocked: .env access is not allowed. API credentials are stored there.",
        file=sys.stderr,
    )
    sys.exit(1)
