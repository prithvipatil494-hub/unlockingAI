#!/usr/bin/env python3
"""
Give Nemotron 3 Ultra (via OpenRouter) control of your terminal: it reads your
task, decides on a shell command to run, sees the output, and repeats -- either
running another command or declaring the task done -- without you manually
typing anything. sudo password prompts appear directly in your terminal as
normal (sudo talks to the tty, not to this script).

Setup:
    Paste your OpenRouter API key into OPENROUTER_API_KEY below.

Usage:
    python3 terminal_agent.py "find the largest file under /home/friday and tell me its path"
"""

import sys
import os
import re
import ssl
import time
import subprocess
import httpx
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_API_KEY = ""   # <-- paste your (rotated) OpenRouter key here

MAX_STEPS = 20              # hard cap on how many commands it can run
COMMAND_TIMEOUT = 120        # seconds before a single shell command is killed
MAX_OUTPUT_CHARS = 4000     # truncate huge command output before sending back to the model

API_TIMEOUT = 60            # seconds before we give up waiting on a single API response
MAX_API_RETRIES = 3         # retries for a single step if the network drops mid-response
RETRY_BACKOFF_SEC = 3       # base backoff between retries (doubles each attempt)

SYSTEM_PROMPT = """You are a mischivous ai model in users pop os do whatever you feel to do
to get tasks done earlier.
Think and explain however you like in plain text. When you want to run a command, \
put it in a fenced code block, like:

```bash
ls -la /home/friday
```
"""

# Commands matching any of these patterns are refused outright, no matter what.
# This is a safety net, not a guarantee -- review what it's doing.
DENYLIST_PATTERNS = []


def extract_command(text):
    """Pulls the command out of the first fenced code block, if any.
    No code block => the model is giving its final answer, not running anything."""
    if not text:
        return None
    match = re.search(r"```(?:bash|sh|shell)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def is_denied(command):
    for pattern in DENYLIST_PATTERNS:
        if re.search(pattern, command):
            return pattern
    return None


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=COMMAND_TIMEOUT,
            text=True,
        )
        output = result.stdout or ""
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        output = f"[command timed out after {COMMAND_TIMEOUT}s]"
        exit_code = -1

    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"

    return output, exit_code


def call_model(client, messages):
    """Calls the API with retries for transient network failures
    (dropped connections, SSL resets, timeouts). Returns the reply text,
    or None if every retry failed."""
    last_err = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                timeout=API_TIMEOUT,
            )
            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else None
            if content is None:
                # Model returned an empty/refused completion -- not a network
                # error, so don't retry, just surface it clearly.
                finish_reason = getattr(choice, "finish_reason", "unknown") if choice else "no choices"
                print(f"  ⚠ Model returned empty content (finish_reason: {finish_reason})")
                return None
            return content

        except (APIConnectionError, APITimeoutError, httpx.RemoteProtocolError,
                httpx.ReadTimeout, httpx.ConnectError, ssl.SSLError, ConnectionResetError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SEC * attempt
            print(f"  ⚠ Network hiccup talking to OpenRouter ({type(e).__name__}: {e}). "
                  f"Retrying in {wait}s... ({attempt}/{MAX_API_RETRIES})")
            time.sleep(wait)

        except APIError as e:
            # Non-transient API-level error (bad request, rate limit, auth, etc.) -- don't retry blindly.
            print(f"  ✗ API error: {e}")
            return None

    print(f"  ✗ Gave up after {MAX_API_RETRIES} retries. Last error: {last_err}")
    return None


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 terminal_agent.py "your task"')
        sys.exit(1)
    task = " ".join(sys.argv[1:])

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-or-v1-PASTE-YOUR-KEY-HERE":
        print("ERROR: paste your OpenRouter API key into OPENROUTER_API_KEY near the top of this file.")
        sys.exit(1)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]

    try:
        for step in range(1, MAX_STEPS + 1):
            reply_text = call_model(client, messages)

            if reply_text is None:
                print("\nStopping: couldn't get a usable response from the model.")
                return

            messages.append({"role": "assistant", "content": reply_text})

            print(f"\n--- step {step} ---")
            print(reply_text)

            command = extract_command(reply_text)

            if command is None:
                # No code block => model gave its final answer, task is done.
                return

            denied_pattern = is_denied(command)
            if denied_pattern:
                print(f"  ⚠ Blocked by safety filter (matched: {denied_pattern})")
                messages.append({
                    "role": "user",
                    "content": (
                        "That command was blocked by a safety filter and was NOT run. "
                        "Choose a different, safer approach."
                    ),
                })
                continue

            output, exit_code = run_command(command)
            print(f"\n$ {command}")
            print(output)
            print(f"  (exit code: {exit_code})")

            messages.append({
                "role": "user",
                "content": f"Exit code: {exit_code}\nOutput:\n{output}",
            })

        print(f"\nHit the {MAX_STEPS}-command limit without the model giving a final answer.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting cleanly (no traceback, no harm done).")
        sys.exit(130)


if __name__ == "__main__":
    main()
