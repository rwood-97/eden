#!/usr/bin/env python3
"""PostToolUse hook: auto-lint and format Python files with Ruff after edits."""

import json
import os
import subprocess
import sys

data = json.load(sys.stdin)
file_path = data.get("tool_input", {}).get("file_path", "")

if file_path.endswith(".py") and os.path.exists(file_path):
    subprocess.run(["uv", "run", "ruff", "check", "--fix", file_path], check=False)
    subprocess.run(["uv", "run", "ruff", "format", file_path], check=False)
