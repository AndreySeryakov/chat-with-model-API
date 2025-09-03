#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Terminal chat with OpenAI or Together (OpenAI-compatible) via Chat Completions.

- Reads system + starter user prompt from a file (single file with markers) or CLI.
- Prints both, then asks you to CONTINUE the starter user prompt before first send.
- Keeps full chat history; supports streaming.
- Loads API keys from .env if present: OPENAI_API_KEY and/or TOGETHER_API_KEY.
- LOGGING: creates a log folder per rules and writes ALL prompts/replies, preserving <think> blocks.

Prompt file format (UTF-8):
    ===SYSTEM===
    <system prompt here>
    ===USER===
    <starter user prompt here>
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

# --- Defaults (change here if you like) --------------------------------------
DEFAULT_PROVIDER = "together"  # "openai" or "together"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
STREAM_BY_DEFAULT = True
# -----------------------------------------------------------------------------

# Try to load .env without hard dependency
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: The 'openai' Python package is required. Install with:\n"
          "    pip install openai python-dotenv\n", file=sys.stderr)
    raise

SYS_HDR = "===SYSTEM==="
USR_HDR = "===USER==="


def parse_prompt_file(path: Path) -> tuple[str, str, str]:
    """Return system, user, and raw file content."""
    text = path.read_text(encoding="utf-8")
    # Accept either our headers or a simple '---' line as a separator
    if SYS_HDR in text and USR_HDR in text:
        _, after_sys = text.split(SYS_HDR, 1)
        system, user = after_sys.split(USR_HDR, 1)
        return system.strip(), user.strip(), text
    # fallback: split on a line with only ---
    parts = re.split(r"(?m)^\s*---\s*$", text)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip(), text
    # fallback: treat whole file as starter user
    return "", text.strip(), text


def build_client(provider: str):
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Missing OPENAI_API_KEY in environment/.env")
        return OpenAI(api_key=api_key), provider
    elif provider == "together":
        # Use OpenAI SDK with Together's OpenAI-compatible base_url
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            sys.exit("Missing TOGETHER_API_KEY in environment/.env")
        client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        return client, provider
    else:
        sys.exit("provider must be 'openai' or 'together'")


def strip_think_blocks(text: str) -> str:
    """Optionally hide <think>...</think> blocks (e.g., DeepSeek R1) from terminal output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# -------- Logging helpers -----------------------------------------------------

def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

def choose_log_dir(prompt_file: str | None) -> Path:
    if prompt_file:
        stem = Path(prompt_file).stem
        return Path(f"{stem}_logs")
    return Path("logs")

def create_log_file(log_dir: Path, provider: str, model: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_s = sanitize_for_filename(model)
    fname = f"{ts}_{provider}_{model_s}.md"
    return log_dir / fname

def write_log_header(log_path: Path, meta: dict, raw_prompt_file: str | None,
                     system_prompt: str, starter_user_full: str):
    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"# Chat Log\n\n")
            f.write(f"- **Datetime:** {meta['datetime']}\n")
            f.write(f"- **Provider:** {meta['provider']}\n")
            f.write(f"- **Model:** {meta['model']}\n")
            f.write(f"- **Temperature:** {meta['temperature']}\n")
            f.write(f"- **Prompt file:** {meta['prompt_file']}\n" if meta['prompt_file'] else "- **Prompt file:** (none)\n")

            f.write(f"- **Streaming:** {meta['stream']}\n")
            f.write(f"- **Strip <think> for terminal:** {meta['strip_think']}\n")
            f.write(f"- **Max tokens per reply:** {meta['max_tokens']}\n\n")

            if raw_prompt_file is not None and meta['prompt_file']:
                f.write("## Prompt file content (raw)\n\n")
                f.write("```text\n")
                f.write(raw_prompt_file)
                f.write("\n```\n\n")

            f.write("## System prompt\n\n")
            f.write("```text\n")
            f.write(system_prompt or "")
            f.write("\n```\n\n")

            f.write("## Starter user prompt (after continuation)\n\n")
            f.write("```text\n")
            f.write(starter_user_full or "")
            f.write("\n```\n\n")

            f.write("---\n\n")
    except Exception as e:
        print(f"[warn] could not write log header: {e}", file=sys.stderr)

def append_log_message(log_path: Path, role: str, content: str):
    """Append a role/content block. Stores RAW content (including <think>)."""
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"### {role}\n\n")
            f.write("```text\n")
            f.write(content or "")
            f.write("\n```\n\n")
    except Exception as e:
        print(f"[warn] could not append to log: {e}", file=sys.stderr)

# -----------------------------------------------------------------------------

def stream_chat_completion(client, model: str, messages: list[dict],
                           temperature: float, max_tokens: int,
                           strip_think: bool) -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    full = []
    try:
        for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                full.append(delta)
                out = strip_think_blocks(delta) if strip_think else delta
                if out:
                    print(out, end="", flush=True)
        print()  # newline after stream
    except KeyboardInterrupt:
        print("\n[stream interrupted]", file=sys.stderr)
    return "".join(full)


def one_shot_completion(client, model: str, messages: list[dict],
                        temperature: float, max_tokens: int,
                        strip_think: bool) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    text = resp.choices[0].message.content or ""
    display = strip_think_blocks(text) if strip_think else text
    print(display)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Terminal chat with OpenAI or Together (OpenAI-compatible)."
    )
    p.add_argument("prompt_file", nargs="?", help="Optional path to prompts file (SYSTEM/USER).")
    p.add_argument("--provider", choices=["openai", "together"], default=DEFAULT_PROVIDER,
                   help=f"API provider (default: {DEFAULT_PROVIDER})")
    p.add_argument("--model", help="Override model name for the selected provider.")
    p.add_argument("--temperature", type=float, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per assistant reply.")
    p.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    p.add_argument("--strip-think", action="store_true",
                   help="Strip <think>...</think> blocks from terminal output (kept in log/history).")
    p.add_argument("--system", help="System prompt text (overrides file).")
    p.add_argument("--user", help="Starter user prompt text (overrides file).")

    args = p.parse_args(argv)

    client, provider = build_client(args.provider)

    model = args.model or (DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_TOGETHER_MODEL)
    temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE
    use_stream = STREAM_BY_DEFAULT and not args.no_stream

    # Gather prompts
    file_system = file_user = ""
    raw_prompt_file = None
    if args.prompt_file:
        file_path = Path(args.prompt_file)
        if not file_path.exists():
            sys.exit(f"Prompt file not found: {file_path}")
        file_system, file_user, raw_prompt_file = parse_prompt_file(file_path)

    system_prompt = (args.system or file_system or "").strip()
    starter_user = (args.user or file_user or "").strip()

    # Show prompts and ask to continue the starter user prompt
    print("=" * 80)
    print("[SYSTEM PROMPT]")
    print(system_prompt if system_prompt else "(empty)")
    print("-" * 80)
    print("[STARTER USER PROMPT]")
    print(starter_user if starter_user else "(empty)")
    print("=" * 80)
    print("You can CONTINUE the starter user prompt now. Type your continuation.")
    print("End with an empty line to send as-is.\n")

    # Multi-line continuation
    continuation_lines = []
    while True:
        line = input("> ")
        if line.strip() == "":
            break
        continuation_lines.append(line)
    if continuation_lines:
        starter_user = (starter_user + "\n" + "\n".join(continuation_lines)).strip()

    # Build initial message history
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if starter_user:
        messages.append({"role": "user", "content": starter_user})
    else:
        # If no starter, ask interactively
        first = input("Enter your first message: ").strip()
        starter_user = first
        messages.append({"role": "user", "content": first})

    # Prepare logging
    log_dir = choose_log_dir(args.prompt_file)
    log_path = create_log_file(log_dir, provider, model)
    meta = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "prompt_file": str(args.prompt_file) if args.prompt_file else "",
        "stream": use_stream,
        "strip_think": args.strip_think,
        "max_tokens": args.max_tokens,
    }
    write_log_header(
        log_path=log_path,
        meta=meta,
        raw_prompt_file=raw_prompt_file,
        system_prompt=system_prompt,
        starter_user_full=starter_user,
    )

    # First assistant reply
    print("\n[assistant]")
    if use_stream:
        assistant_raw = stream_chat_completion(
            client, model, messages, temperature, args.max_tokens, args.strip_think
        )
    else:
        assistant_raw = one_shot_completion(
            client, model, messages, temperature, args.max_tokens, args.strip_think
        )
    # Log first turn (store RAW assistant text incl. <think> if any)
    append_log_message(log_path, "assistant", assistant_raw)
    messages.append({"role": "assistant", "content": assistant_raw})

    # Chat loop
    print(f"\nLog file: {log_path}")
    print("Type '/exit' to quit.")
    while True:
        try:
            user_text = input("\n[you] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_text})
        append_log_message(log_path, "user", user_text)

        print("\n[assistant]")
        if use_stream:
            assistant_raw = stream_chat_completion(
                client, model, messages, temperature, args.max_tokens, args.strip_think
            )
        else:
            assistant_raw = one_shot_completion(
                client, model, messages, temperature, args.max_tokens, args.strip_think
            )
        messages.append({"role": "assistant", "content": assistant_raw})
        append_log_message(log_path, "assistant", assistant_raw)


if __name__ == "__main__":
    main()
