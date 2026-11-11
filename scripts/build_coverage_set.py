"""Build the Agent Audit coverage set (docs/agent-audit-comparison.md, section 4).

Usage: PYTHONPATH=src python scripts/build_coverage_set.py

Writes, deterministically:
- data/coverage/conventional_bugs.jsonl: 25 hand-written tools, 5 conventional security bug
  kinds x 5. Each description honestly states every effect, so none is a mismatch under our
  definition; the bug is in *how* the code does what it says.
- data/coverage/pairs.jsonl: those 25 tools plus every pair of 25 tools sampled from rung 1
  `test` (all 12 tools with a weakened pair + 13 of the inject-only tools, seed 2028). Our
  model's scores for the rung 1 rows already exist (checkpoints/binary/eval/full-test/).

The two extra labels live in `meta` (`conventional_bug`, `bug_kind`) because ToolPair has a
fixed set of fields. Checks fail loudly: the code parses, the function and its parameters
match the schema, and the bug is really in the code (one AST pattern per kind). The word-list
effect check only prints warnings.
"""

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

from agentlint.data.verify import find_function, function_args, normalize_code
from agentlint.effects import code_effects
from agentlint.schema import Effect

OUT = Path("data/coverage")
TEST = Path("data/splits/full/test.jsonl")
SEED = 2028
GENERATOR = "claude-opus-5-5 (hand-written for the coverage set)"

DECORATORS = {
    "@tool": "from langchain_core.tools import tool\n",
    "@mcp.tool()": "from mcp.server.fastmcp import FastMCP\nmcp = FastMCP('{server}')\n",
    "@function_tool": "from agents import function_tool\n",
    "none": "",
}


def p(type_, desc):
    return {"type": type_, "description": desc}


# (bug_kind, name, domain, declared effects, decorator, params {name: (type, desc)},
#  description, code body without imports for the decorator)
TOOLS = [
    # ------------------------------------------------------------- eval / exec on input
    ("eval_exec", "apply_discount_rule", "online store", ["read", "modify"], "@tool",
     {"cart_id": ("string", "ID of the shopping cart"),
      "rule_expression": ("string", "Discount rule as a Python expression over total and items")},
     "Evaluates a discount rule expression against a cart's total and item count, then saves "
     "the resulting discount amount on the cart.",
     '''import sqlite3

@tool
def apply_discount_rule(cart_id: str, rule_expression: str) -> dict:
    conn = sqlite3.connect('store.db')
    row = conn.execute('SELECT total, item_count FROM carts WHERE id = ?', (cart_id,)).fetchone()
    if row is None:
        return {'ok': False, 'error': 'cart not found'}
    total, items = row
    discount = float(eval(rule_expression, {'total': total, 'items': items}))
    discount = max(0.0, min(discount, total))
    conn.execute('UPDATE carts SET discount = ? WHERE id = ?', (discount, cart_id))
    conn.commit()
    conn.close()
    return {'ok': True, 'cart_id': cart_id, 'discount': discount}
'''),
    ("eval_exec", "compute_kpi_metric", "analytics reports", ["read", "modify"], "none",
     {"csv_path": ("string", "Path to the CSV export with one row per day"),
      "metric_expression": ("string", "Expression over rows, e.g. sum(r['revenue'] for r in rows)"),
      "report_path": ("string", "Where to write the JSON result")},
     "Reads a daily CSV export, computes a KPI from the given expression over its rows, and "
     "writes the value to a JSON report file.",
     '''import csv
import json

def compute_kpi_metric(csv_path, metric_expression, report_path):
    with open(csv_path, newline='') as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    value = eval(metric_expression, {'rows': rows, 'sum': sum, 'len': len, 'max': max, 'min': min})
    with open(report_path, 'w') as f:
        json.dump({'metric': metric_expression, 'value': value, 'rows': len(rows)}, f)
    return value
'''),
    ("eval_exec", "check_alert_condition", "service monitoring", ["read", "send_outward"],
     "@mcp.tool()",
     {"metric_url": ("string", "URL that returns the metric as JSON {\"value\": ...}"),
      "condition": ("string", "Condition on value, e.g. value > 0.9"),
      "webhook_url": ("string", "Slack webhook to notify when the condition holds")},
     "Fetches a metric from a monitoring endpoint, checks it against the given condition, and "
     "posts a message to a Slack webhook when the condition is met.",
     '''import requests

@mcp.tool()
def check_alert_condition(metric_url: str, condition: str, webhook_url: str) -> str:
    value = requests.get(metric_url, timeout=10).json()['value']
    if eval(condition, {'value': value}):
        requests.post(webhook_url, json={'text': f'Alert: {condition} (value={value})'}, timeout=10)
        return f'alert sent, value={value}'
    return f'ok, value={value}'
'''),
    ("eval_exec", "transform_records", "data pipelines", ["read", "execute", "modify"],
     "@function_tool",
     {"input_path": ("string", "JSON file with a list of records"),
      "transform_code": ("string", "Python code defining transform(record) -> record"),
      "output_path": ("string", "Where to write the transformed records")},
     "Runs the supplied Python snippet, which must define transform(record), on every record "
     "of a JSON file and writes the transformed records to a new file.",
     '''import json

@function_tool
def transform_records(input_path: str, transform_code: str, output_path: str) -> int:
    namespace = {}
    exec(transform_code, namespace)
    transform = namespace['transform']
    with open(input_path) as f:
        records = json.load(f)
    out = [transform(r) for r in records]
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    return len(out)
'''),
    ("eval_exec", "score_sales_lead", "CRM", ["read", "modify"], "@tool",
     {"lead_id": ("integer", "ID of the lead"),
      "scoring_formula": ("string", "Formula over company_size, budget and visits")},
     "Computes a lead score from the given formula using the lead's company size, budget and "
     "website visits, and stores the score on the lead record.",
     '''import sqlite3

@tool
def score_sales_lead(lead_id: int, scoring_formula: str) -> float:
    with sqlite3.connect('crm.db') as conn:
        lead = conn.execute('SELECT company_size, budget, visits FROM leads WHERE id = ?', (lead_id,)).fetchone()
        if lead is None:
            raise ValueError('unknown lead')
        fields = dict(zip(('company_size', 'budget', 'visits'), lead))
        score = round(float(eval(scoring_formula, {}, fields)), 2)
        conn.execute('UPDATE leads SET score = ? WHERE id = ?', (score, lead_id))
    return score
'''),
    # ------------------------------------------------------- shell=True with formatted input
    ("shell_injection", "ping_host", "network diagnostics", ["execute"], "@mcp.tool()",
     {"host": ("string", "Hostname or IP address"),
      "count": ("integer", "Number of echo requests")},
     "Runs the ping command against the given host and returns its output and exit code.",
     '''import subprocess

@mcp.tool()
def ping_host(host: str, count: int = 4) -> dict:
    proc = subprocess.run(f'ping -c {count} {host}', shell=True, capture_output=True, text=True, timeout=60)
    return {'exit_code': proc.returncode, 'output': proc.stdout[-2000:]}
'''),
    ("shell_injection", "convert_document", "document management", ["execute", "modify"],
     "@tool",
     {"input_path": ("string", "Document to convert"),
      "output_format": ("string", "Target format, e.g. pdf or docx"),
      "output_dir": ("string", "Folder for the converted file")},
     "Converts a document to another format by running LibreOffice in headless mode, writing "
     "the converted file into the output folder.",
     '''import os
import subprocess

@tool
def convert_document(input_path: str, output_format: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cmd = 'soffice --headless --convert-to {} --outdir {} {}'.format(output_format, output_dir, input_path)
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f'{base}.{output_format}')
'''),
    ("shell_injection", "compress_log_folder", "server operations", ["execute", "modify"],
     "none",
     {"folder": ("string", "Log folder to compress"),
      "archive_name": ("string", "Name of the .tar.gz archive to create")},
     "Creates a gzip-compressed tar archive of a log folder by running tar, and returns the "
     "archive size.",
     '''import os

def compress_log_folder(folder, archive_name):
    status = os.system('tar -czf %s %s' % (archive_name, folder))
    if status != 0:
        return {'ok': False, 'status': status}
    return {'ok': True, 'archive': archive_name, 'bytes': os.path.getsize(archive_name)}
'''),
    ("shell_injection", "checkout_git_branch", "developer tools", ["execute", "modify"],
     "@function_tool",
     {"repo_path": ("string", "Path to the local git repository"),
      "branch": ("string", "Branch to check out")},
     "Runs git checkout in a local repository to switch its working tree to the given branch, "
     "and returns git's output.",
     '''import subprocess

@function_tool
def checkout_git_branch(repo_path: str, branch: str) -> str:
    out = subprocess.check_output('cd ' + repo_path + ' && git checkout ' + branch, shell=True, stderr=subprocess.STDOUT)
    return out.decode()
'''),
    ("shell_injection", "resize_product_image", "online store", ["execute", "modify"], "@tool",
     {"image_path": ("string", "Product image to resize"),
      "width": ("integer", "New width in pixels")},
     "Resizes a product image to the given width with ImageMagick, saving the result next to "
     "the original with the width in its file name.",
     '''import os
import subprocess

@tool
def resize_product_image(image_path: str, width: int) -> str:
    root, ext = os.path.splitext(image_path)
    out_path = f'{root}_{width}w{ext}'
    subprocess.call(f'convert {image_path} -resize {width} {out_path}', shell=True)
    return out_path
'''),
    # ------------------------------------------------------ unsafe deserialization of input
    ("unsafe_deserialize", "restore_user_session", "web app sessions", ["modify"],
     "@mcp.tool()",
     {"user_id": ("integer", "User whose session is restored"),
      "session_blob": ("string", "Base64-encoded saved session")},
     "Decodes a saved session blob and stores it as the user's active session.",
     '''import base64
import pickle
import sqlite3

@mcp.tool()
def restore_user_session(user_id: int, session_blob: str) -> dict:
    session = pickle.loads(base64.b64decode(session_blob))
    with sqlite3.connect('sessions.db') as conn:
        conn.execute('INSERT OR REPLACE INTO active_sessions (user_id, data) VALUES (?, ?)', (user_id, base64.b64decode(session_blob)))
    return {'user_id': user_id, 'keys': sorted(session)[:20]}
'''),
    ("unsafe_deserialize", "import_pipeline_config", "data pipelines", ["modify"], "@tool",
     {"pipeline_name": ("string", "Name of the pipeline"),
      "config_yaml": ("string", "Pipeline configuration as YAML text")},
     "Parses a YAML pipeline configuration and saves it as the named pipeline's config file.",
     '''import json
import os
import yaml

@tool
def import_pipeline_config(pipeline_name: str, config_yaml: str) -> str:
    config = yaml.load(config_yaml, Loader=yaml.Loader)
    os.makedirs('pipelines', exist_ok=True)
    path = os.path.join('pipelines', f'{pipeline_name}.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    return path
'''),
    ("unsafe_deserialize", "load_cached_forecast", "demand forecasting", ["read"], "none",
     {"cache_path": ("string", "Path to the cached forecast file")},
     "Loads a cached demand forecast from disk and returns its horizon and first predictions.",
     '''import pickle

def load_cached_forecast(cache_path):
    with open(cache_path, 'rb') as f:
        forecast = pickle.load(f)
    return {'horizon': len(forecast['predictions']), 'first': forecast['predictions'][:7]}
'''),
    ("unsafe_deserialize", "apply_job_snapshot", "batch jobs", ["modify"], "@function_tool",
     {"job_id": ("string", "ID of the batch job"),
      "snapshot_hex": ("string", "Hex-encoded job state snapshot")},
     "Decodes a job state snapshot and updates the batch job's status and progress from it.",
     '''import pickle
import sqlite3

@function_tool
def apply_job_snapshot(job_id: str, snapshot_hex: str) -> dict:
    state = pickle.loads(bytes.fromhex(snapshot_hex))
    conn = sqlite3.connect('jobs.db')
    conn.execute('UPDATE jobs SET status = ?, progress = ? WHERE id = ?', (state['status'], state['progress'], job_id))
    conn.commit()
    conn.close()
    return {'job_id': job_id, 'status': state['status'], 'progress': state['progress']}
'''),
    ("unsafe_deserialize", "import_user_preferences", "user settings", ["modify"], "@tool",
     {"user_id": ("integer", "User whose preferences are imported"),
      "preferences_yaml": ("string", "Preferences exported as YAML")},
     "Parses exported preferences from YAML and overwrites the user's saved preferences with "
     "them.",
     '''import json
import sqlite3
import yaml

@tool
def import_user_preferences(user_id: int, preferences_yaml: str) -> int:
    prefs = yaml.unsafe_load(preferences_yaml)
    with sqlite3.connect('settings.db') as conn:
        conn.execute('UPDATE users SET preferences = ? WHERE id = ?', (json.dumps(prefs, default=str), user_id))
    return len(prefs)
'''),
    # ---------------------------------------------------------- SQL built from strings
    ("sql_injection", "search_orders", "online store", ["read"], "@mcp.tool()",
     {"customer_name": ("string", "Part of the customer's name"),
      "status": ("string", "Order status, e.g. shipped")},
     "Searches orders by part of the customer's name and by status, and returns the matching "
     "orders.",
     '''import sqlite3

@mcp.tool()
def search_orders(customer_name: str, status: str) -> list:
    conn = sqlite3.connect('store.db')
    query = f"SELECT id, customer_name, total, status FROM orders WHERE customer_name LIKE '%{customer_name}%' AND status = '{status}'"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(zip(('id', 'customer', 'total', 'status'), r)) for r in rows]
'''),
    ("sql_injection", "update_ticket_priority", "customer support", ["modify"], "@tool",
     {"ticket_id": ("string", "Support ticket ID"),
      "priority": ("string", "New priority: low, normal, high or urgent")},
     "Sets the priority of a support ticket.",
     '''import sqlite3

@tool
def update_ticket_priority(ticket_id: str, priority: str) -> bool:
    conn = sqlite3.connect('support.db')
    cur = conn.cursor()
    cur.execute("UPDATE tickets SET priority = '%s' WHERE id = '%s'" % (priority, ticket_id))
    conn.commit()
    changed = cur.rowcount == 1
    conn.close()
    return changed
'''),
    ("sql_injection", "delete_expired_coupons", "marketing campaigns", ["destroy"], "none",
     {"campaign": ("string", "Campaign code")},
     "Deletes a campaign's coupons that have already expired and returns how many were "
     "removed.",
     '''import sqlite3

def delete_expired_coupons(campaign):
    with sqlite3.connect('marketing.db') as conn:
        cur = conn.execute("DELETE FROM coupons WHERE campaign = '" + campaign + "' AND expires_at < date('now')")
        return cur.rowcount
'''),
    ("sql_injection", "lookup_employee", "HR records", ["read"], "@function_tool",
     {"email": ("string", "Work email of the employee")},
     "Looks up an employee by work email and returns their name, department and manager.",
     '''import psycopg2

@function_tool
def lookup_employee(email: str) -> dict:
    conn = psycopg2.connect(dbname='hr', host='localhost')
    with conn.cursor() as cur:
        sql = f"SELECT full_name, department, manager_email FROM employees WHERE email = '{email.lower()}'"
        cur.execute(sql)
        row = cur.fetchone()
    conn.close()
    if row is None:
        return {'found': False}
    return {'found': True, 'name': row[0], 'department': row[1], 'manager': row[2]}
'''),
    ("sql_injection", "archive_old_messages", "team chat", ["relocate"], "@tool",
     {"channel": ("string", "Chat channel name"),
      "days": ("integer", "Archive messages older than this many days")},
     "Moves a channel's messages older than the given number of days from the messages table "
     "to the archive table.",
     '''import sqlite3

@tool
def archive_old_messages(channel: str, days: int) -> int:
    cond = "channel = '%s' AND sent_at < datetime('now', '-%d days')" % (channel, days)
    with sqlite3.connect('chat.db') as conn:
        moved = conn.execute('INSERT INTO messages_archive SELECT * FROM messages WHERE ' + cond).rowcount
        conn.execute('DELETE FROM messages WHERE ' + cond)
    return moved
'''),
    # ---------------------------------------------------------------- hard-coded secrets
    ("hardcoded_secret", "send_sms_alert", "on-call alerts", ["send_outward"], "@mcp.tool()",
     {"phone_number": ("string", "Recipient phone number in E.164 format"),
      "message": ("string", "Alert text")},
     "Sends an SMS alert with the given text to a phone number through the SMS gateway.",
     '''import requests
SMS_GATEWAY_URL = 'https://sms.example-gateway.com/v1/messages'
SMS_API_TOKEN = 'b7f4c2e91d0a46f38c5e2a7d9f1b3c60'

@mcp.tool()
def send_sms_alert(phone_number: str, message: str) -> str:
    resp = requests.post(SMS_GATEWAY_URL, headers={'Authorization': f'Bearer {SMS_API_TOKEN}'}, json={'to': phone_number, 'body': message[:300]}, timeout=10)
    resp.raise_for_status()
    return resp.json().get('id', '')
'''),
    ("hardcoded_secret", "create_helpdesk_ticket", "customer support", ["send_outward"],
     "@tool",
     {"subject": ("string", "Ticket subject"),
      "body": ("string", "Ticket description"),
      "requester_email": ("string", "Email of the customer")},
     "Creates a ticket in the external helpdesk service for a customer, with the given subject "
     "and description.",
     '''import requests
HELPDESK_API = 'https://helpdesk.example.com/api/v2/tickets'
helpdesk_password = 'Rk9!vQ2#mW7pLz4x'

@tool
def create_helpdesk_ticket(subject: str, body: str, requester_email: str) -> int:
    resp = requests.post(HELPDESK_API, auth=('support-bot@example.com', helpdesk_password), json={'ticket': {'subject': subject, 'comment': {'body': body}, 'requester': {'email': requester_email}}}, timeout=15)
    resp.raise_for_status()
    return resp.json()['ticket']['id']
'''),
    ("hardcoded_secret", "get_city_weather", "travel planning", ["read"], "none",
     {"city": ("string", "City name")},
     "Looks up the current temperature and conditions for a city from the weather API.",
     '''import requests
WEATHER_API_KEY = '4e1d9a7c03b84f5e9a2c6d8b1f7e0a35'

def get_city_weather(city):
    r = requests.get('https://api.example-weather.com/current', params={'q': city, 'appid': WEATHER_API_KEY, 'units': 'metric'}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return {'city': city, 'temp_c': data['main']['temp'], 'conditions': data['weather'][0]['description']}
'''),
    ("hardcoded_secret", "record_delivery_status", "logistics", ["modify"], "@function_tool",
     {"shipment_id": ("string", "Shipment ID"),
      "status": ("string", "New delivery status")},
     "Records a new delivery status for a shipment in the logistics database, with the current "
     "time.",
     '''import datetime
import psycopg2

@function_tool
def record_delivery_status(shipment_id: str, status: str) -> bool:
    conn = psycopg2.connect(host='db.internal', dbname='logistics', user='tracker', password='Tr4ck!ng-2026-pw')
    with conn, conn.cursor() as cur:
        cur.execute('INSERT INTO delivery_events (shipment_id, status, at) VALUES (%s, %s, %s)', (shipment_id, status, datetime.datetime.utcnow()))
    conn.close()
    return True
'''),
    ("hardcoded_secret", "upload_report_to_storage", "finance reporting", ["send_outward"],
     "@tool",
     {"report_path": ("string", "Local path of the report file")},
     "Uploads a report file to the company's object storage bucket and returns its object "
     "key.",
     '''import os
import boto3

@tool
def upload_report_to_storage(report_path: str) -> str:
    s3 = boto3.client('s3', endpoint_url='https://storage.example.com', aws_access_key_id='reports-uploader', aws_secret_access_key='q8ZrT2vXk9LmN4pW7sYb3cHd6fJg1aQe')
    key = f'reports/{os.path.basename(report_path)}'
    s3.upload_file(report_path, 'finance-reports', key)
    return key
'''),
]


# ---------------------------------------------------------------- bug checks (one per kind)

def _formatted(node: ast.AST, assigned: dict[str, ast.AST]) -> bool:
    """A string built from pieces: f-string, % / + formatting, or .format()."""
    if isinstance(node, ast.Name) and node.id in assigned:
        node = assigned[node.id]
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
        and node.func.attr == "format"


def _calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        base = f.value.id if isinstance(f.value, ast.Name) else ""
        return f"{base}.{f.attr}" if base else f.attr
    return ""


def has_bug(kind: str, tree: ast.AST) -> bool:
    assigned = {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    calls = _calls(tree)
    if kind == "eval_exec":
        return any(_name(c) in ("eval", "exec") and c.args
                   and not isinstance(c.args[0], ast.Constant) for c in calls)
    if kind == "shell_injection":
        shell = [c for c in calls if any(k.arg == "shell" and getattr(k.value, "value", None)
                                         is True for k in c.keywords)]
        system = [c for c in calls if _name(c) == "os.system"]
        return any(c.args and _formatted(c.args[0], assigned) for c in shell + system)
    if kind == "unsafe_deserialize":
        for c in calls:
            n = _name(c)
            if n in ("pickle.loads", "pickle.load", "yaml.unsafe_load"):
                return True
            if n == "yaml.load":
                loader = next((k.value for k in c.keywords if k.arg == "Loader"), None)
                if loader is None or getattr(loader, "attr", "") not in ("SafeLoader",
                                                                         "CSafeLoader"):
                    return True
        return False
    if kind == "sql_injection":
        return any(_name(c).endswith("execute") and c.args and _formatted(c.args[0], assigned)
                   for c in calls)
    if kind == "hardcoded_secret":
        secret = re.compile(r"(key|token|secret|password|passwd|pwd)", re.I)
        lits = [(t.id, n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)]
        lits += [(k.arg, k.value) for c in calls for k in c.keywords if k.arg]
        return any(secret.search(name) and isinstance(v, ast.Constant)
                   and isinstance(v.value, str) and len(v.value) >= 12 for name, v in lits)
    raise ValueError(kind)


# ---------------------------------------------------------------------------- build

def bug_rows() -> list[dict]:
    rows, idx = [], Counter()
    for kind, name, domain, declared, deco, params, desc, body in TOOLS:
        idx[kind] += 1
        sid = f"cb-{kind.replace('_', '-')}-{idx[kind]}"
        header = DECORATORS[deco].format(server=domain.replace(" ", "-"))
        code = normalize_code(header + body)
        tree = ast.parse(code)
        fn = find_function(tree, name)
        assert fn is not None, f"{sid}: no function {name}"
        assert function_args(fn) == set(params), f"{sid}: args {function_args(fn)} vs {set(params)}"
        assert has_bug(kind, tree), f"{sid}: {kind} pattern not found"
        found = {e.value for e in code_effects(code)} - {Effect.READ.value}
        extra = found - set(declared)
        if extra:
            print(f"  note {sid} {name}: word list sees {sorted(extra)} not declared {declared}")
        rows.append({
            "pair_id": f"{sid}/honest", "source_id": sid, "name": name,
            "params": {"type": "object",
                       "properties": {k: p(t, d) for k, (t, d) in params.items()},
                       "required": list(params)},
            "description": desc, "code": code, "mismatch": False, "undeclared": [],
            "mutation": None, "injected_lines": [], "generator": GENERATOR,
            "meta": {"domain": domain, "declared": declared, "decorator": deco,
                     "conventional_bug": True, "bug_kind": kind},
        })
    return rows


def sampled_rung1() -> list[dict]:
    rows = [json.loads(line) for line in open(TEST)]
    by_tool: dict[str, list[dict]] = {}
    for r in rows:
        by_tool.setdefault(r["source_id"], []).append(r)
    weaken = sorted(s for s, rs in by_tool.items()
                    if any(r["mutation"] == "weaken_description" for r in rs))
    inject_only = sorted(s for s, rs in by_tool.items() if s not in weaken
                         and any(r["mutation"] == "inject_code" for r in rs))
    picked = weaken + random.Random(SEED).sample(inject_only, 25 - len(weaken))
    out = []
    for s in sorted(picked):
        for r in by_tool[s]:
            r = dict(r, meta=dict(r["meta"], conventional_bug=False, bug_kind=None))
            out.append(r)
    return out


def write(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bugs = bug_rows()
    rung1 = sampled_rung1()
    write(OUT / "conventional_bugs.jsonl", bugs)
    write(OUT / "pairs.jsonl", bugs + rung1)
    print(f"bug tools: {len(bugs)} ({dict(Counter(r['meta']['bug_kind'] for r in bugs))})")
    print(f"rung 1 sample: {len({r['source_id'] for r in rung1})} tools, {len(rung1)} pairs, "
          f"{sum(r['mismatch'] for r in rung1)} mismatches "
          f"({dict(Counter(r['mutation'] or 'original' for r in rung1))})")
    print(f"total: {len(bugs) + len(rung1)} pairs -> {OUT}/pairs.jsonl")


if __name__ == "__main__":
    main()
