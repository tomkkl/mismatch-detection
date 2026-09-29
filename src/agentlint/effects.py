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
                     "terminate", "deactivate", "close"),
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


def call_effects(call: ast.Call) -> set[Effect]:
    name = dotted_name(call)
    if not name:
        return set()
    last = name.rsplit(".", 1)[-1]
    head = name.split(".", 1)[0]

    # SQL through a DB-API cursor: classify by the statement, not the method.
    if last in ("execute", "executemany", "executescript"):
        sql = _first_str_arg(call)
        if sql is not None:
            words = tokens(sql)
            return {_SQL[words[0]]} if words and words[0] in _SQL else set()
    if head == "subprocess" or name in ("os.system", "os.popen") or name.startswith("os.exec"):
        return {Effect.EXECUTE}
    if name in ("shutil.move", "os.rename", "os.replace", "os.renames"):
        return {Effect.RELOCATE}
    if name in ("shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree"):
        return {Effect.MODIFY}
    if name in ("requests.post", "requests.put", "requests.patch", "httpx.post", "httpx.put"):
        return {Effect.SEND_OUTWARD}
    if name in ("requests.delete", "httpx.delete"):
        return {Effect.SEND_OUTWARD, Effect.DESTROY}
    if last == "open":
        mode = next((kw.value.value for kw in call.keywords
                     if kw.arg == "mode" and isinstance(kw.value, ast.Constant)), None)
        if mode is None and len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            mode = call.args[1].value
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return {Effect.MODIFY}
        return {Effect.READ}

    words = tokens(last)
    if not words:
        return set()
    return {e for e, verbs in _CALL_VERBS.items() if words[0] in verbs or last in verbs}


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


def code_effects(code: str | ast.AST) -> dict[Effect, list[int]]:
    """Map each effect the lexicon sees in `code` to the lines where it appears."""
    tree = ast.parse(code) if isinstance(code, str) else code
    found: dict[Effect, set[int]] = {}
    for node in ast.walk(tree):
        effects = call_effects(node) if isinstance(node, ast.Call) else _assign_effects(node)
        for e in effects:
            found.setdefault(e, set()).add(node.lineno)
    return {e: sorted(lines) for e, lines in found.items()}