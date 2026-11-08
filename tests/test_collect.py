"""Rung 3 collection: extraction from real-looking tool files, and the collect run with a
fake GitHub client and a fake LLM (no network, no CLI calls)."""

import json

from agentlint.data.collect import collect, extract_tools, split_docstring
from agentlint.data.verify import verify_honest
from agentlint.schema import ToolPair

SOURCE = '''
import os
import shutil
from typing import Annotated, Optional

import requests
from langchain_core.tools import tool
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("files")
API_URL = os.environ.get("WEATHER_URL", "https://weather.example.com")
UNUSED = 3


def _fetch(city):
    return requests.get(f"{API_URL}/now", params={"q": city}, timeout=10).json()


def _unrelated():
    return shutil.disk_usage("/")


@tool
def current_weather(city: str, units: Optional[str] = "metric") -> dict:
    """Look up the current weather for a city.

    Args:
        city: Name of the city to look up.
        units: Unit system, metric or imperial.
    """
    return _fetch(city)


@mcp.tool()
def archive_report(path: Annotated[str, "Report file to archive"], keep: bool = False) -> str:
    """Moves a finished report into the archive folder and returns its new path."""
    return shutil.move(path, "archive/")


@mcp.tool()
def no_doc(x: int) -> int:
    return abs(x)


@tool
def chinese(query: str) -> str:
    """搜索功能，返回搜索结果的摘要文本。搜索功能，返回搜索结果"""
    return _fetch(query)


def not_a_tool(a):
    """Plain function with a long enough docstring to pass the length check."""
    return current_weather(a)
'''


def _by_name(tools):
    return {t["name"]: t for t in tools}


def test_extract_tools_keeps_function_and_what_it_uses():
    tools, skipped = extract_tools(SOURCE)
    got = _by_name(tools)
    assert set(got) == {"current_weather", "archive_report"}
    assert any("no_doc: no docstring" in s for s in skipped)
    assert any("chinese: description not in English" in s for s in skipped)

    w = got["current_weather"]
    assert w["framework"] == "langchain"
    assert w["description"] == "Look up the current weather for a city."
    assert w["params"] == {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string", "description": "Name of the city to look up."},
        "units": {"type": "string", "description": "Unit system, metric or imperial."}}}
    # The helper, the constant it reads and the imports both use come along; nothing else.
    assert "def _fetch" in w["code"] and "API_URL =" in w["code"] and "import os" in w["code"]
    assert "import requests" in w["code"] and "from langchain_core.tools import tool" in w["code"]
    for absent in ("_unrelated", "UNUSED", "archive_report", "not_a_tool", "FastMCP",
                   '"""', "Args:"):
        assert absent not in w["code"]

    a = got["archive_report"]
    assert a["framework"] == "mcp"
    assert a["params"]["properties"]["path"]["description"] == "Report file to archive"
    assert a["params"]["properties"]["keep"]["type"] == "boolean"
    assert a["params"]["required"] == ["path"]
    assert "mcp = FastMCP('files')" in a["code"] and "import shutil" in a["code"]
    assert "def _fetch" not in a["code"]


def test_extracted_tool_passes_the_honest_check():
    tools, _ = extract_tools(SOURCE)
    for raw in tools:
        t = ToolPair(pair_id="g/honest", source_id="g", name=raw["name"], params=raw["params"],
                     description=raw["description"], code=raw["code"], mismatch=False,
                     meta={"declared": ["read"]})
        assert verify_honest(t) == (True, "ok")


def test_split_docstring_sphinx_and_sections():
    desc, params = split_docstring(
        "Send a message to a Slack channel.\n\n:param channel: Channel id.\n"
        ":param str text: Message text.\nReturns:\n    The message timestamp.")
    assert params == {"channel": "Channel id.", "text": "Message text."}
    assert desc == "Send a message to a Slack channel. Returns: The message timestamp."
    desc, params = split_docstring(
        "Fetch prices from start to end.\n- symbol (str): Ticker.\n- note: not a param.",
        {"symbol"})
    assert params == {"symbol": "Ticker."}
    assert desc == "Fetch prices from start to end. - note: not a param."


class FakeGitHub:
    def __init__(self, files):
        self.files = files  # path -> source

    def search(self, query, pages):
        fw = "mcp" if "mcp" in query else "agents" if "function_tool" in query else "lc"
        return [{"repository": {"full_name": f"org/{fw}"}, "path": p,
                 "url": f"https://api.github.com/repos/org/{fw}/contents/{p}?ref=abc123"}
                for p in self.files]

    def file(self, item):
        return self.files[item["path"]], "abc123"

    def license(self, repo):
        return {"org/lc": "MIT", "org/mcp": None}.get(repo, "NOASSERTION")


class FakeLabelLLM:
    name = "fake"

    def json(self, prompt, schema, system, effort=None):
        honest = "archive_report" not in prompt
        return {"domain": "weather data", "declared": ["read", "send_outward", "read"],
                "honest": honest, "reason": "" if honest else "the move is not mentioned"}


def test_collect_writes_reviewable_candidates(tmp_path):
    known = tmp_path / "train.jsonl"
    known.write_text(json.dumps({"name": "archive_report"}) + "\n")
    cfg = {"generator": {"effort": "low"}, "collect": {
        "out": str(tmp_path / "rung3" / "candidates.jsonl"), "target": 10,
        "queries": {"langchain": "@tool", "mcp": "@mcp.tool"},
        "exclude_names_from": [str(known)]}}
    gh = FakeGitHub({"agent/tools.py": SOURCE, "tests/test_tools.py": SOURCE})
    stats = collect(cfg, FakeLabelLLM(), gh, log=lambda s: None)

    rows = [json.loads(l) for l in open(cfg["collect"]["out"], encoding="utf-8")]
    # One tool: the test file is skipped, archive_report is a known name, and the copy in
    # the unlicensed repo (second query) is never read.
    assert [r["name"] for r in rows] == ["current_weather"] and stats["tools"] == 1
    skipped = (tmp_path / "rung3" / "collect_skipped.txt").read_text()
    assert "org/mcp/agent/tools.py: license none" in skipped
    r = rows[0]
    assert r["source_id"] == "g0000" and r["generator"] == "github" and not r["mismatch"]
    assert r["meta"]["source"] == {
        "repo": "org/lc", "path": "agent/tools.py", "commit": "abc123", "license": "MIT",
        "url": "https://github.com/org/lc/blob/abc123/agent/tools.py"}
    assert r["meta"]["declared"] == ["read", "send_outward"]
    assert r["meta"]["domain"] == "weather data" and r["meta"]["draft"]["honest"]
    assert r["meta"]["review"] is None  # a person decides
    assert (tmp_path / "rung3" / "collect_skipped.txt").exists()
