"""Invoke the model endpoints already configured on this host."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


@dataclass
class Reply:
    text: str
    returned_model: str | None
    usage: dict
    finish_reason: str | None = None


class ModelError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def invoke_glm(prompt):
    path = Path.home() / ".claude" / "settings.json"
    settings = json.loads(path.read_text())["env"]
    base = settings["ANTHROPIC_BASE_URL"].rstrip("/")
    parts = urlsplit(base)
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ModelError("Invalid GLM gateway URL")
    endpoint = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    headers = {
        "Authorization": "Bearer " + settings["ANTHROPIC_AUTH_TOKEN"],
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 400,
        "thinking": {"type": "disabled"},
    }
    request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=90) as response:
            data = json.load(response)
    except HTTPError as error:
        raise ModelError(f"GLM HTTP {error.code}") from None
    if data.get("model") != "glm-5.3":
        raise ModelError("GLM response model ID did not match glm-5.3")
    answer = "\n".join(block["text"] for block in data["content"] if block["type"] == "text")
    return Reply(answer, data["model"], data.get("usage", {}), data.get("stop_reason"))


def invoke_cli(command, prompt, directory):
    process = subprocess.run(command, input=prompt, capture_output=True, text=True,
                             cwd=directory, timeout=120)
    if process.returncode:
        output = (process.stdout + process.stderr).lower()
        markers = ("401", "403", "404", "429", "quota", "rate limit", "model_not_found",
                   "unauthorized", "timed out", "not supported", "unexpected argument")
        matched = ", ".join(marker for marker in markers if marker in output)
        raise ModelError(f"CLI exited {process.returncode}" + (f" ({matched})" if matched else ""))
    return process.stdout


def invoke_deepseek(prompt):
    launcher = Path.home() / ".local" / "bin" / "claude-ds"
    with TemporaryDirectory(prefix="graphdo-ds-") as directory:
        output = invoke_cli([
            "bash", "-ic", 'exec "$@"', "graphdo-deepseek", str(launcher),
            "--print", "--tools", "", "--disable-slash-commands", "--no-session-persistence",
            "--output-format", "json", "--model", "deepseek-v4.1-flash",
        ], prompt, directory)
    data = json.loads(output)
    if data.get("is_error"):
        raise ModelError("Claude CLI returned an error")
    model = next(iter(data.get("modelUsage") or {}), None)
    if model != "deepseek-v4.1-flash":
        raise ModelError("Claude CLI did not report deepseek-v4.1-flash")
    return Reply(data["result"], model, data.get("usage", {}), data.get("stop_reason"))


def invoke_luna(prompt):
    with TemporaryDirectory(prefix="graphdo-luna-") as directory:
        output = invoke_cli([
            "codex", "-a", "never", "exec", "--ignore-user-config", "--skip-git-repo-check",
            "--ephemeral", "--json",
            "--sandbox", "read-only", "-C", directory, "-m", "gpt-6-luna",
            "-c", 'model_reasoning_effort="low"', "-c", 'web_search="disabled"',
            "-c", "features.shell_tool=false", "-c", "features.apps=false",
            "-c", "features.multi_agent=false", "-",
        ], prompt, directory)
    messages = []
    usage = None
    reported_model = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        reported_model = event.get("model") or reported_model
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            messages.append(item["text"])
        if event.get("type") == "turn.completed":
            usage = event.get("usage", {})
    if usage is None or not messages:
        raise ModelError("Codex did not emit a completed answer")
    if reported_model and reported_model != "gpt-6-luna":
        raise ModelError("Codex event reported a different model ID")
    return Reply(messages[-1], reported_model, usage)


ADAPTERS = {
    "deepseek-v4.1-flash": invoke_deepseek,
    "glm-5.3": invoke_glm,
    "gpt-6-luna": invoke_luna,
}
