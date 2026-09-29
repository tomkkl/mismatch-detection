"""LLM access for dataset generation.

Generation runs through the `claude` CLI in print mode, so it draws on the user's Claude plan
usage rather than the paid API. Every answer is cached on disk by a hash of the request, so a
run that stops (usage limit, crash, Ctrl+C) resumes where it left off.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Protocol


class LLMError(RuntimeError):
    pass


class UsageLimitError(LLMError):
    """The plan's usage window is used up. Stop the run and resume later."""


class LLM(Protocol):
    name: str

    def json(self, prompt: str, schema: dict, system: str, effort: str | None = None) -> dict:
        """Return an object matching `schema`."""


class ClaudeCLI:
    def __init__(self, model: str = "opus", effort: str = "medium",
                 cache_dir: str | Path = "data/generated/llm_cache", timeout: int = 900):
        exe = shutil.which("claude")
        if exe is None:
            raise LLMError("`claude` CLI not found on PATH")
        self.exe = exe
        self.model = model
        self.effort = effort
        self.name = f"claude-cli:{model}"
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Run outside the repo so the project's CLAUDE.md never reaches the generator.
        self.workdir = Path(tempfile.gettempdir()) / "agentlint_llm"
        self.workdir.mkdir(exist_ok=True)
        # Without an API key in the environment the CLI uses the logged-in plan.
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.cost_usd = 0.0  # list-price equivalent reported by the CLI, not a charge

    def json(self, prompt: str, schema: dict, system: str, effort: str | None = None) -> dict:
        effort = effort or self.effort
        key = hashlib.sha256(json.dumps(
            [self.model, effort, system, prompt, schema], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with self._lock:
                self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["output"]

        cmd = [self.exe, "-p", "--model", self.model, "--effort", effort,
               "--output-format", "json", "--json-schema", json.dumps(schema),
               "--system-prompt", system, "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--disable-slash-commands"]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  encoding="utf-8", cwd=self.workdir, env=self.env,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude CLI timed out after {self.timeout}s") from e

        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            text = (proc.stdout + proc.stderr).strip()
            if "limit" in text.lower():
                raise UsageLimitError(text[:500])
            raise LLMError(f"claude CLI exit {proc.returncode}: {text[:500]}")

        if result.get("is_error") or result.get("structured_output") is None:
            text = str(result.get("result", ""))
            if "limit" in text.lower() or result.get("api_error_status") == 429:
                raise UsageLimitError(text[:500])
            raise LLMError(f"claude CLI error ({result.get('subtype')}): {text[:500]}")

        output = result["structured_output"]
        cost = float(result.get("total_cost_usd") or 0.0)
        path.write_text(json.dumps({"output": output, "cost_usd": cost}), encoding="utf-8")
        with self._lock:
            self.calls += 1
            self.cost_usd += cost
        return output