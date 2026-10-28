#!/usr/bin/env python3
"""Query the locally configured GLM, DeepSeek, and Codex-compatible APIs.

Examples:
    python3 query_llms.py --list
    python3 query_llms.py --profile claude-glm --prompt "Hello"
    python3 query_llms.py --profile claude-ds --prompt "Summarize this" --max-tokens 512
    python3 query_llms.py --profile codex --prompt "Explain this code" --json
    python3 query_llms.py --all --prompt "Give one sentence about graph reasoning"

The script reads credentials from the same local files as the launchers:
  - claude-glm: ~/.codex/provider-secrets.env (SUB2API_API_KEY)
  - claude-ds:  ~/.codex/deepseek-secrets.env (DEEPSEEK_API_KEY)
  - codex:      ~/.codex/auth.json (OPENAI_API_KEY)

Credentials can also be supplied through the corresponding environment
variables. They are never printed or written by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CODEX_HOME = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


@dataclass(frozen=True)
class Profile:
    name: str
    api: str
    base_url: str
    model: str
    secret_file: Optional[Path]
    secret_name: str
    description: str

    @property
    def endpoint(self) -> str:
        if self.api == "anthropic":
            return self.base_url.rstrip("/") + "/v1/messages"
        # The local Codex config uses the gateway root as base_url; Codex's
        # responses wire appends /responses (rather than adding another /v1).
        return self.base_url.rstrip("/") + "/responses"


PROFILES = {
    "claude-glm": Profile(
        name="claude-glm",
        api="anthropic",
        base_url=os.environ.get("GLM_BASE_URL", "http://3.81.169.29:8080"),
        model=os.environ.get("GLM_MODEL", "glm-5.3"),
        secret_file=CODEX_HOME / "provider-secrets.env",
        secret_name="SUB2API_API_KEY",
        description="Claude-compatible gateway serving GLM-5.3",
    ),
    "claude-ds": Profile(
        name="claude-ds",
        api="anthropic",
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "http://3.81.169.29:8080"),
        model=os.environ.get("CLAUDE_DS_MODEL", "deepseek-v4.1-flash"),
        secret_file=CODEX_HOME / "deepseek-secrets.env",
        secret_name="DEEPSEEK_API_KEY",
        description="Claude-compatible gateway serving DeepSeek V4.1 Flash",
    ),
    "codex": Profile(
        name="codex",
        api="responses",
        base_url=os.environ.get("CODEX_BASE_URL", "http://35.173.184.233:8080"),
        model=os.environ.get("CODEX_MODEL", "gpt-6-astra"),
        secret_file=CODEX_HOME / "auth.json",
        secret_name="OPENAI_API_KEY",
        description="OpenAI Responses-compatible Codex gateway",
    ),
}

ALIASES = {"glm": "claude-glm", "ds": "claude-ds", "deepseek": "claude-ds"}


def _env_file_value(path: Path, name: str) -> Optional[str]:
    """Read one export assignment without sourcing or executing the file."""
    if not path.is_file():
        return None
    assignment = re.compile(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=\s*(.*?)\s*$")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = assignment.match(line)
            if not match:
                continue
            value = match.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value
    except OSError:
        return None
    return None


def _secret_for(profile: Profile) -> Optional[str]:
    # Environment variables take precedence, which makes the script portable.
    value = os.environ.get(profile.secret_name)
    if value:
        return value
    if profile.secret_file is None or not profile.secret_file.is_file():
        return None
    if profile.secret_file.suffix == ".json":
        try:
            data = json.loads(profile.secret_file.read_text(encoding="utf-8"))
            value = data.get(profile.secret_name)
            return value if isinstance(value, str) and value else None
        except (OSError, ValueError):
            return None
    return _env_file_value(profile.secret_file, profile.secret_name)


def _content_text(value: Any) -> str:
    """Extract text from common Anthropic/OpenAI response content shapes."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def _extract_text(data: Dict[str, Any], api: str) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    if api == "anthropic":
        text = _content_text(data.get("content"))
        if text:
            return text
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", choices[0])
        if isinstance(message, dict):
            text = _content_text(message.get("content", message.get("text", "")))
            if text:
                return text
    output = data.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            if isinstance(item, dict):
                chunks.append(_content_text(item.get("content", item.get("text", ""))))
        text = "".join(chunks)
        if text:
            return text
    return ""


def _request_json(profile: Profile, prompt: str, secret: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
    if profile.api == "anthropic":
        payload = {
            "model": profile.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {secret}",
            "x-api-key": secret,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        payload = {
            "model": profile.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        }

    request = Request(
        profile.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        # Do not echo the response body: some gateways include request metadata.
        raise RuntimeError(f"{profile.name}: HTTP {exc.code} from {profile.endpoint}") from exc
    except URLError as exc:
        raise RuntimeError(f"{profile.name}: connection failed ({exc.reason})") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{profile.name}: gateway returned non-JSON data") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{profile.name}: gateway returned an unexpected JSON shape")
    if data.get("error"):
        error = data["error"]
        message = error.get("message", "unknown API error") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"{profile.name}: {message}")
    return data


def query(profile_name: str, prompt: str, *, max_tokens: int = 1024, timeout: float = 120.0,
          model: Optional[str] = None) -> Dict[str, Any]:
    canonical = ALIASES.get(profile_name, profile_name)
    if canonical not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; choose from {', '.join(PROFILES)}")
    profile = PROFILES[canonical]
    if model:
        profile = Profile(**{**profile.__dict__, "model": model})
    secret = _secret_for(profile)
    if not secret:
        raise RuntimeError(f"{profile.name}: credential {profile.secret_name} is not configured")
    raw = _request_json(profile, prompt, secret, max_tokens, timeout)
    text = _extract_text(raw, profile.api)
    if not text:
        raise RuntimeError(f"{profile.name}: could not find text in the API response")
    return {"profile": profile.name, "model": profile.model, "text": text, "response": raw}


def _print_profiles() -> None:
    for profile in PROFILES.values():
        configured = "yes" if _secret_for(profile) else "no"
        print(f"{profile.name:12} model={profile.model:22} credential={configured:3}  {profile.description}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted((*PROFILES.keys(), *ALIASES.keys())), help="one API profile")
    parser.add_argument("--all", action="store_true", help="query all three profiles")
    parser.add_argument("--prompt", help="prompt text; omit to read UTF-8 text from stdin")
    parser.add_argument("--model", help="override the configured model for this request")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="print structured JSON including the raw response")
    parser.add_argument("--list", action="store_true", help="list profiles and credential availability")
    args = parser.parse_args(argv)

    if args.list:
        _print_profiles()
        return 0
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--max-tokens must be positive and --timeout must be greater than zero")
    if bool(args.profile) == bool(args.all):
        parser.error("choose exactly one of --profile or --all")
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        parser.error("provide --prompt or pipe a nonempty prompt on stdin")

    names = list(PROFILES) if args.all else [args.profile]
    results = []
    failures = []
    for name in names:
        try:
            results.append(query(name, prompt, max_tokens=args.max_tokens, timeout=args.timeout, model=args.model))
        except (RuntimeError, ValueError) as exc:
            failures.append({"profile": name, "error": str(exc)})

    if args.json:
        print(json.dumps({"results": results, "errors": failures}, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"=== {result['profile']} ({result['model']}) ===")
            print(result["text"])
        for failure in failures:
            print(f"=== {failure['profile']} ERROR ===\n{failure['error']}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
