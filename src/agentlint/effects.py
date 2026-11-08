"""Effect vocabulary shared by label checks and localization.

Two lexicons:
- text: word stems that *declare* an effect in a name, parameter, or description.
- code: call names that *perform* an effect.

Both are deliberately rough. They are used as label hygiene (leak checks, soft agreement
signals) and to point at lines, never as the detector itself.
"""

import ast
import re

from agentlint.schema import Effect

# Plain definitions used in generation prompts.
DEFINITIONS: dict[Effect, str] = {
    Effect.READ: "reads data without changing it (files, database rows, API lookups)",
    Effect.MODIFY: "changes stored data or creates new records or files",
    Effect.DESTROY: "deletes, cancels, or irreversibly removes data or objects",
    Effect.RELOCATE: "moves or renames data to a different place (files, folders, records "
                     "moved to another table or bucket)",
    Effect.SEND_OUTWARD: "sends data outside the system (email, HTTP call to another service, "
                         "upload, message queue, webhook)",
    Effect.EXECUTE: "runs a program, shell command, script, or job",
    Effect.SPEND: "moves money (charge, refund, payment, purchase, payout, credit)",
    Effect.CHANGE_AUTHORITY: "changes who can do what (roles, permissions, ownership, access "
                             "tokens, sharing, invites)",
}

# Ideas offered to the injector so injected ops stay ordinary and fit the tool's domain.
INJECTION_IDEAS: dict[Effect, str] = {
    Effect.READ: "read an extra file, table, or record the tool does not need to mention",
    Effect.MODIFY: "update a status or timestamp field, overwrite a setting, write a cache "
                   "file, increment a counter",
    Effect.DESTROY: "delete a record, remove the source or a temp file, cancel an order or "
                    "subscription, drop old rows",
    Effect.RELOCATE: "move the file or folder somewhere else, rename it, move rows to an "
                     "archive table, move an object to another bucket",
    Effect.SEND_OUTWARD: "POST the data to another internal service, email a copy, upload it "
                         "to storage, publish it to a queue",
    Effect.EXECUTE: "run a maintenance command or script with subprocess, trigger a build "
                    "or job runner",
    Effect.SPEND: "issue a refund, charge a stored payment method, create a payout, apply "
                  "an account credit",
    Effect.CHANGE_AUTHORITY: "grant a role, add a user as collaborator, chmod a file, change "
                             "an owner, create an access token",
}

# ---------------------------------------------------------------------------- text

# Matched as a prefix of a word token ("delet" matches delete / deletes / deletion).
_TEXT_STEMS: dict[Effect, tuple[str, ...]] = {
    Effect.READ: ("read", "fetch", "retriev", "lookup", "view", "load", "search", "inspect"),
    Effect.MODIFY: ("updat", "modif", "edit", "writ", "overwrit", "sav", "append", "insert",
                    "patch", "mark", "increment", "decrement", "adjust", "rewrit"),
    Effect.DESTROY: ("delet", "remov", "purg", "eras", "destroy", "wip", "drop", "truncat",
                     "cancel", "discard", "prun", "unlink", "clean"),
    Effect.RELOCATE: ("move", "moving", "reloc", "archiv", "migrat", "renam"),
    Effect.SEND_OUTWARD: ("send", "sent", "email", "mail", "notif", "upload", "publish",
                          "shar", "export", "webhook", "forward", "broadcast", "transmit",
                          "slack", "sms", "tweet"),
    Effect.EXECUTE: ("execut", "exec", "launch", "spawn", "invok", "shell", "command",
                     "script", "subprocess", "eval"),
    Effect.SPEND: ("pay", "charg", "refund", "purchas", "buy", "bill", "invoic", "transact",
                   "credit", "debit", "payout", "transfer", "fund"),
    Effect.CHANGE_AUTHORITY: ("grant", "revok", "permiss", "role", "admin", "owner",
                              "access", "privileg", "authoriz", "invit", "promot", "demot",
                              "chmod", "chown", "acl"),
}

# Short words matched as whole tokens only.
_TEXT_EXACT: dict[Effect, tuple[str, ...]] = {
    Effect.READ: ("get", "list", "show", "find", "query"),
    Effect.MODIFY: ("set", "add", "put", "store"),
    Effect.DESTROY: ("rm", "clear"),
    Effect.RELOCATE: ("mv",),
    Effect.SEND_OUTWARD: ("post", "push", "sync"),
    Effect.EXECUTE: ("run", "runs", "running", "ran", "exec"),
    Effect.SPEND: ("pay", "order"),
    Effect.CHANGE_AUTHORITY: ("sudo",),
}


def tokens(text: str) -> list[str]:
    """Split prose or identifiers (snake_case, camelCase) into lowercase word tokens."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def text_mentions(text: str, effect: Effect) -> bool:
    """True if `text` contains a word that declares `effect`."""
    stems, exact = _TEXT_STEMS[effect], _TEXT_EXACT[effect]
    return any(t in exact or t.startswith(stems) for t in tokens(text))


# ---------------------------------------------------------------------------- code

# Keyed by the first word of the called function's last name segment
# (`db.delete_rows(...)` -> "delete").
_CALL_VERBS: dict[Effect, tuple[str, ...]] = {
    Effect.READ: ("read", "get", "fetch", "fetchone", "fetchall", "load", "loads", "list",
                  "find", "search", "query", "select", "listdir", "iterdir", "glob", "scandir",
                  "walk", "stat", "retrieve", "lookup", "download"),
    Effect.MODIFY: ("write", "update", "save", "set", "put", "patch", "insert", "append",
                    "mark", "edit", "modify", "increment", "decrement", "upsert", "touch",
                    "mkdir", "makedirs", "setdefault", "dump", "dumps", "add", "commit"),
    Effect.DESTROY: ("remove", "unlink", "rmdir", "rmtree", "delete", "drop", "truncate",
                     "purge", "cancel", "destroy", "wipe", "discard", "clear", "void",
                     "terminate", "deactivate"),  # bare close() is not destroy (see below)
    Effect.RELOCATE: ("move", "rename", "relocate", "archive", "migrate"),
    Effect.SEND_OUTWARD: ("post", "send", "sendmail", "send_message", "publish", "upload",
                          "notify", "urlopen", "emit", "forward", "broadcast", "email",
                          "webhook", "export", "sync", "share", "tweet"),
    Effect.EXECUTE: ("popen", "system", "check_call", "check_output", "execv", "execvp",
                     "spawn", "launch", "exec", "run_command", "startfile"),
    Effect.SPEND: ("charge", "refund", "pay", "purchase", "capture", "payout", "transfer",
                   "buy", "bill", "invoice", "debit", "credit", "withdraw"),
    Effect.CHANGE_AUTHORITY: ("chmod", "chown", "grant", "revoke", "assign", "promote",
                              "demote", "invite", "authorize", "elevate", "setuid"),
}

_SQL = {
    "select": Effect.READ, "update": Effect.MODIFY, "insert": Effect.MODIFY,
    "replace": Effect.MODIFY, "delete": Effect.DESTROY, "drop": Effect.DESTROY,
    "truncate": Effect.DESTROY, "grant": Effect.CHANGE_AUTHORITY,
    "revoke": Effect.CHANGE_AUTHORITY,
}

# Table / URL / call-name nouns (2026-11-07, from the word-list misses on `full` and rung 2):
# writing to a money table or endpoint moves money; writing to an access table changes who
# may do what. Matched as a prefix of a word token, like the text stems.
_SPEND_NOUNS = ("payment", "payout", "charge", "refund", "credit", "invoice", "billing",
                "transaction", "wage", "fee", "royalt", "bount", "stipend", "deposit",
                "payroll", "debit", "transfer")
_AUTHORITY_NOUNS = ("role", "permission", "access", "acl", "share", "collaborator",
                    "member", "grant", "invit", "token", "privileg", "polic", "owner",
                    "login_link", "login")
_ARCHIVE_NOUNS = ("archive", "archived", "history", "purged", "backup", "trash")
_SPEND_SDK = ("PaymentIntent", "Refund", "Transfer", "Charge", "Payout", "Invoice",
              "CreditNote", "Topup")
_HTTP_SEND = {"post", "put", "patch", "delete", "request", "urlopen", "Request"}

# Attribute or key names whose assignment changes who may do what.
_AUTHORITY_FIELDS = ("role", "roles", "admin", "is_admin", "is_staff", "is_superuser",
                     "permission", "permissions", "owner", "owner_id", "scope", "scopes",
                     "access_level", "privileges")


def dotted_name(node: ast.AST) -> str:
    """`a.b.c(...)` -> "a.b.c"; unknown shapes -> ""."""
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    if call.args and isinstance(call.args[0], ast.JoinedStr):
        return "".join(v.value for v in call.args[0].values
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _has(words: list[str], nouns: tuple[str, ...]) -> bool:
    return any(w.startswith(nouns) for w in words)


def _table_words(name: str) -> list[str]:
    """`billing.account_credits` -> ["billing", "account", "credits"]."""
    return tokens(name.replace(".", "_"))


# A table is only an archive copy when its own name says so ("lots_archive").
_SQL_INSERT = re.compile(r"\binsert\s+(?:or\s+\w+\s+)?into\s+([\w.]+)(.*)", re.I | re.S)
_SQL_UPDATE = re.compile(r"\bupdate\s+([\w.]+)\s+set\s+(.*?)(?:\bwhere\b|$)", re.I | re.S)
_SQL_DELETE = re.compile(r"\bdelete\s+from\s+([\w.]+)", re.I)
_SQL_AUTH_COLUMNS = ("role", "access", "permission", "revoked", "can_", "is_admin",
                     "is_staff", "owner", "scope", "privileg")


def sql_effects(sql: str) -> set[Effect]:
    """Effects of an SQL string: the statement verb, plus what the touched tables are for
    (money, access, archive copies). Handles several statements and `WITH ... DELETE`."""
    out: set[Effect] = set()
    for stmt in sql.split(";"):
        words = tokens(stmt)
        if not words:
            continue
        verbs = [w for w in words if w in _SQL]
        if words[0] in _SQL:
            out.add(_SQL[words[0]])
        if words[0] == "create":
            out.add(Effect.MODIFY)
        if "delete" in verbs and words[0] == "with":
            out.add(Effect.DESTROY)
        m = _SQL_INSERT.search(stmt)
        if m:
            table, rest = _table_words(m.group(1)), m.group(2)
            out.add(Effect.MODIFY)
            if re.search(r"\bselect\b", rest, re.I) and _has(table, _ARCHIVE_NOUNS):
                out.add(Effect.RELOCATE)
            if re.search(r"\bselect\b", rest, re.I) and re.search(r"\bdelete\b", stmt, re.I):
                out.add(Effect.RELOCATE)  # WITH moved AS (DELETE ... RETURNING) INSERT ...
            if _has(table, _AUTHORITY_NOUNS):  # invoice_grants: access, not money
                out.add(Effect.CHANGE_AUTHORITY)
            elif _has(table, _SPEND_NOUNS):
                out.add(Effect.SPEND)
        m = _SQL_UPDATE.search(stmt)
        if m:
            table, cols = _table_words(m.group(1)), m.group(2)
            set_cols = [c.lower() for c in re.findall(r"(\w+)\s*=", cols)]
            out.add(Effect.MODIFY)
            if _has(table, _AUTHORITY_NOUNS) or any(c.startswith(_SQL_AUTH_COLUMNS)
                                                     for c in set_cols):
                out.add(Effect.CHANGE_AUTHORITY)
            if any(c.startswith(("balance", "credit_balance", "credit_cents")) for c in set_cols):
                out.add(Effect.SPEND)
        m = _SQL_DELETE.search(stmt)
        if m and _has(_table_words(m.group(1)), _AUTHORITY_NOUNS):
            out.add(Effect.CHANGE_AUTHORITY)
    return out


_WRITE_VERBS = ("create", "attach", "put", "add", "set", "update", "delete", "remove",
                "detach", "revoke", "grant", "assign", "modify", "issue", "apply", "sync",
                "ensure", "restrict", "release", "give", "enable", "disable", "settle",
                "process", "record", "insert", "post", "send", "transfer", "charge", "refund",
                "pay", "credit", "debit")


def _string_text(node: ast.AST | None, consts: dict[str, str]) -> str:
    """Literal text of a URL / command argument: constants, f-string parts, `a + b`,
    module-level string constants by name, and list elements."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return " ".join(_string_text(v.value if isinstance(v, ast.FormattedValue) else v, consts)
                        for v in node.values)
    if isinstance(node, ast.BinOp):
        return _string_text(node.left, consts) + " " + _string_text(node.right, consts)
    if isinstance(node, ast.Name):
        return consts.get(node.id, "")
    if isinstance(node, (ast.List, ast.Tuple)):
        return " ".join(_string_text(e, consts) for e in node.elts)
    if isinstance(node, ast.Call):  # os.environ.get('X', 'default'), url.rstrip('/') + ...
        return " ".join(_string_text(a, consts) for a in node.args) + " " + \
            _string_text(node.func.value if isinstance(node.func, ast.Attribute) else None, consts)
    return ""


def _payload_has_amount(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg in ("json", "data") and isinstance(kw.value, ast.Dict):
            keys = [k.value for k in kw.value.keys if isinstance(k, ast.Constant)]
            if any(isinstance(k, str) and k.startswith("amount") for k in keys):
                return True
    return False


def call_effects(call: ast.Call, consts: dict[str, str] | None = None) -> set[Effect]:
    consts = consts or {}
    name = dotted_name(call)
    if not name:
        return set()
    last = name.rsplit(".", 1)[-1]
    head = name.split(".", 1)[0]
    parts = name.split(".")

    # SQL through a DB-API cursor: classify by the statement and its tables, not the method.
    if last in ("execute", "executemany", "executescript"):
        sql = _first_str_arg(call)
        if sql is not None:
            return sql_effects(sql)
    if head == "subprocess" or name in ("os.system", "os.popen") or name.startswith("os.exec"):
        out = {Effect.EXECUTE}
        args = set(tokens(" ".join(_string_text(a, consts) for a in call.args)))
        if args & {"grant", "revoke", "setfacl", "chmod", "chown", "role", "rolebinding",
                   "grant_access"} or any(a.startswith("grant") for a in args):
            out.add(Effect.CHANGE_AUTHORITY)
        if args & {"refund", "charge", "payout"}:
            out.add(Effect.SPEND)
        if args & {"delete", "rm", "purge", "destroy"}:
            out.add(Effect.DESTROY)
        return out
    if name in ("shutil.move", "os.rename", "os.replace", "os.renames"):
        return {Effect.RELOCATE}
    if name in ("shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree"):
        return {Effect.MODIFY}
    if last == "copy_object":  # object storage: copy to a new key (with a delete, a move)
        return {Effect.RELOCATE}
    if any(p in _SPEND_SDK for p in parts) and last in ("create", "capture", "confirm", "pay",
                                                         "finalize_invoice"):
        return {Effect.SPEND}  # as our honest tools declare it: the payment, not the call
    if last[:1].isupper():  # constructors (TransferResult(...)): no effect of their own
        return set()

    out: set[Effect] = set()
    # HTTP sends: what the endpoint or payload is for (billing, access).
    if last in _HTTP_SEND or name.endswith("urlopen"):
        url = tokens(_string_text(call.args[0] if call.args else None, consts) + " " +
                     " ".join(_string_text(kw.value, consts) for kw in call.keywords
                              if kw.arg in ("url",)))
        authority = _has(url, _AUTHORITY_NOUNS) and last != "get"
        if authority:
            out.add(Effect.CHANGE_AUTHORITY)
        if last in ("post", "put", "patch", "Request", "request", "urlopen") and (
                _payload_has_amount(call) or (_has(url, _SPEND_NOUNS) and not authority)):
            out.add(Effect.SPEND)
    if name in ("requests.post", "requests.put", "requests.patch", "httpx.post", "httpx.put",
                "httpx.patch"):
        out.add(Effect.SEND_OUTWARD)
        if last in ("put", "patch"):
            out.add(Effect.MODIFY)
        return out
    if name in ("requests.delete", "httpx.delete"):
        return out | {Effect.SEND_OUTWARD, Effect.DESTROY}

    # Helpers and SDK methods named for what they do: `_issue_supplier_refund`,
    # `attach_group_policy`, `create_login_link`.
    words = tokens(last)
    if words and words[0] in _WRITE_VERBS:
        if _has(words[1:], _SPEND_NOUNS):
            out.add(Effect.SPEND)
        if _has(words[1:], _AUTHORITY_NOUNS) or "login_link" in last:
            out.add(Effect.CHANGE_AUTHORITY)
    if last == "open":
        mode = next((kw.value.value for kw in call.keywords
                     if kw.arg == "mode" and isinstance(kw.value, ast.Constant)), None)
        if mode is None and len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            mode = call.args[1].value
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return out | {Effect.MODIFY}
        return out | {Effect.READ}

    if not words:
        return out
    if words[0] == "close" and len(words) > 1:  # close_account, close_ticket; not conn.close()
        out.add(Effect.DESTROY)
    return out | {e for e, verbs in _CALL_VERBS.items() if words[0] in verbs or last in verbs}


def _assign_effects(node: ast.AST) -> set[Effect]:
    targets = []
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    out = set()
    for t in targets:
        field = None
        if isinstance(t, ast.Attribute):
            field = t.attr
        elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
            field = str(t.slice.value)
        if field is None:
            continue
        out.add(Effect.CHANGE_AUTHORITY if field.lower() in _AUTHORITY_FIELDS else Effect.MODIFY)
    return out


def _module_strings(tree: ast.AST) -> dict[str, str]:
    """Names bound to string-ish values anywhere in the code (`BILLING_URL = '...'`,
    `os.environ.get('X', '...')`, f-strings), so URLs can be read where they are used."""
    consts: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            text = _string_text(node.value, consts)
            if text.strip():
                consts[node.targets[0].id] = text
    return consts


def code_effects(code: str | ast.AST) -> dict[Effect, list[int]]:
    """Map each effect the lexicon sees in `code` to the lines where it appears."""
    tree = ast.parse(code) if isinstance(code, str) else code
    consts = _module_strings(tree)
    found: dict[Effect, set[int]] = {}
    for node in ast.walk(tree):
        effects = call_effects(node, consts) if isinstance(node, ast.Call) \
            else _assign_effects(node)
        for e in effects:
            found.setdefault(e, set()).add(node.lineno)
    return {e: sorted(lines) for e, lines in found.items()}