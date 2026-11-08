"""Coverage set: bug checks, Agent Audit case files and result parsing, the report."""

from pathlib import Path

import pytest

from agentlint.data.bugs import BUG_TYPES, has_bug
from agentlint.eval.agent_audit import findings_by_pair, with_docstring, write_cases
from agentlint.eval.coverage import report
from agentlint.schema import ToolPair

BUGGY = {
    "eval_exec": "def calc(expr):\n    return eval(expr)",
    "shell_injection": "import subprocess\ndef ping(host):\n"
                       "    return subprocess.run(f'ping -c 1 {host}', shell=True)",
    "sql_injection": "import sqlite3\ndef find(name):\n    c = sqlite3.connect('a.db')\n"
                     "    return c.execute(f\"SELECT * FROM users WHERE name = '{name}'\").fetchall()",
    "unsafe_deserialization": "import pickle\ndef load(blob):\n    return pickle.loads(blob)",
    "hardcoded_secret": "import requests\nAPI_KEY = 'demo-key-0123456789abcdef'\ndef get(q):\n"
                        "    return requests.get('https://x.io', params={'q': q, 'k': API_KEY})",
    "path_traversal": "import os\ndef read(name):\n"
                      "    return open(os.path.join('/srv/docs', name)).read()",
    "ssrf": "import requests\ndef fetch(url):\n    return requests.get(url, timeout=5).text",
}
SAFE = {
    "eval_exec": "import ast\ndef calc(expr):\n    return ast.literal_eval(expr)",
    "shell_injection": "import subprocess\ndef ping(host):\n"
                       "    return subprocess.run(['ping', '-c', '1', host])",
    "sql_injection": "import sqlite3\ndef find(name):\n    c = sqlite3.connect('a.db')\n"
                     "    return c.execute('SELECT * FROM users WHERE name = ?', (name,)).fetchall()",
    "unsafe_deserialization": "import json\ndef load(blob):\n    return json.loads(blob)",
    "hardcoded_secret": "import os, requests\nAPI_KEY = os.environ['API_KEY']\ndef get(q):\n"
                        "    return requests.get('https://x.io', params={'q': q, 'k': API_KEY})",
    "path_traversal": "import os\ndef read(name):\n    p = os.path.realpath(os.path.join('/srv', name))\n"
                      "    assert p.startswith('/srv/')\n    return open(p).read()",
    "ssrf": "import requests\ndef fetch(path):\n"
            "    return requests.get('https://api.example.com/' + path, timeout=5).text",
}
NAMES = {"eval_exec": "calc", "shell_injection": "ping", "sql_injection": "find",
         "unsafe_deserialization": "load", "hardcoded_secret": "get",
         "path_traversal": "read", "ssrf": "fetch"}


@pytest.mark.parametrize("bug", list(BUG_TYPES))
def test_has_bug_finds_bug_and_not_safe_version(bug):
    assert has_bug(BUGGY[bug], NAMES[bug], bug)
    assert not has_bug(SAFE[bug], NAMES[bug], bug)


def _pair(pid, name="calc", code="def calc(expr):\n    return eval(expr)", **kw):
    return ToolPair(pair_id=pid, source_id=pid.split("/")[0], name=name, params={},
                    description="Evaluates a math expression.", code=code, mismatch=False, **kw)


def test_cases_get_neutral_paths_and_docstrings(tmp_path):
    root = tmp_path / "coverage"
    cases = write_cases([_pair("a/honest"), _pair("b/honest")], root)
    assert list(cases) == ["case_0000/calc_service/tools/calc.py",
                           "case_0001/calc_service/tools/calc.py"]
    text = (root / "case_0000/calc_service/tools/calc.py").read_text()
    assert '"""Evaluates a math expression."""' in text or "'Evaluates a math expression.'" in text
    assert with_docstring(_pair("x/honest")).index("Evaluates") < \
        with_docstring(_pair("x/honest")).index("eval(")


def test_findings_by_pair_skips_suppressed_and_keeps_empty_cases():
    root = Path("probes/aa_run/coverage")
    cases = {"case_0000/calc_service/tools/calc.py": "a/honest",
             "case_0001/calc_service/tools/calc.py": "b/honest"}
    loc = lambda c: {"file_path": f"probes/aa_run/coverage/{c}/calc_service/tools/calc.py",
                     "start_line": 2}
    result = {"findings": [
        {"rule_id": "AGENT-017", "severity": "critical", "tier": "BLOCK", "suppressed": False,
         "location": loc("case_0000")},
        {"rule_id": "AGENT-022", "severity": "medium", "tier": "INFO", "suppressed": True,
         "location": loc("case_0001")}]}
    got = findings_by_pair(result, cases, root)
    assert [f["rule_id"] for f in got["a/honest"]] == ["AGENT-017"] and got["b/honest"] == []


def test_report_groups_and_counts():
    pairs = [
        {"pair_id": "m/inject", "mismatch": True, "mutation": "inject_code", "meta": {}},
        {"pair_id": "h/honest", "mismatch": False, "mutation": None, "meta": {}},
        {"pair_id": "b/honest", "mismatch": False, "mutation": None,
         "meta": {"bug_type": "eval_exec"}},
    ]
    findings = {"b/honest": [{"rule_id": "AGENT-017"}], "h/honest": [], "m/inject": []}
    scores = {"m/inject": 0.99, "h/honest": 0.01, "b/honest": 0.2}
    t = report(pairs, findings, scores)
    assert t["mismatch: inject"]["model"] == 1 and t["mismatch: inject"]["agent_audit"] == 0
    assert t["bug: eval_exec"]["agent_audit"] == 1 and t["bug: eval_exec"]["either"] == 1
    assert t["bug: eval_exec"]["top_rules"] == {"AGENT-017": 1}
    assert t["honest: original"]["either"] == 0
