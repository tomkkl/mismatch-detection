"""Word-list forms added 2026-11-07 from the misses on `full` and rung 2, and the forms that
must stay quiet (honest tools were over-flagged by early versions)."""

import pytest

from agentlint.effects import code_effects, sql_effects
from agentlint.localize import locate
from agentlint.schema import Effect

S, A, R, D, M = (Effect.SPEND, Effect.CHANGE_AUTHORITY, Effect.RELOCATE, Effect.DESTROY,
                 Effect.MODIFY)


@pytest.mark.parametrize("sql, want", [
    ("INSERT INTO lots_archive SELECT * FROM lots WHERE sku = ?", {M, R}),
    ("WITH moved AS (DELETE FROM reqs WHERE id = %s RETURNING *) "
     "INSERT INTO reqs_archive SELECT * FROM moved", {D, M, R}),
    ("INSERT INTO payouts (user_id, amount) VALUES (?, ?)", {M, S}),
    ("INSERT INTO account_credits (account_id, amount_cents) VALUES (?, ?)", {M, S}),
    ("UPDATE author_accounts SET balance_cents = balance_cents + ? WHERE id = ?", {M, S}),
    ("INSERT OR IGNORE INTO order_permissions (order_id, role) VALUES (?, ?)", {M, A}),
    ("INSERT OR REPLACE INTO invoice_grants (id, user_id) VALUES (?, ?)", {M, A}),
    ("DELETE FROM deal_collaborators WHERE deal_id = ?", {D, A}),
    ("UPDATE deal_permissions SET access_level = 'read_only' WHERE deal_id = ?", {M, A}),
    ("UPDATE telehealth_tokens SET revoked_at = ? WHERE appointment_id = ?", {M, A}),
    ("SELECT id, amount FROM payments WHERE user_id = ?", {Effect.READ}),
    ("INSERT INTO notes (body) SELECT body FROM drafts WHERE id = ?", {M}),
    ("CREATE TABLE IF NOT EXISTS runs_archive AS SELECT * FROM runs WHERE 0", {M}),
])
def test_sql_effects(sql, want):
    assert sql_effects(sql) == want


def _fx(code: str) -> set[Effect]:
    return set(code_effects(code))


def test_http_to_billing_endpoints_spends():
    assert S in _fx("import requests, os\nURL = os.environ.get('BILLING_URL', "
                    "'https://billing.internal/api/charges')\ndef f(a):\n"
                    "    requests.post(URL, json={'account': a})")
    assert S in _fx("import requests\ndef f(a):\n"
                    "    requests.post(f'{BASE}/accounts/{a}/credits', json={'note': 'x'})")
    assert S in _fx("import requests\ndef f(a):\n"
                    "    requests.post('https://x.io/v1/fees', json={'amount_cents': 5})")
    # An access endpoint on a billing service is access, not money.
    fx = _fx("import requests\ndef f(i, e):\n    requests.post(f'{BILLING}/invoices/{i}/access', "
             "json={'email': e, 'role': 'payer'})")
    assert A in fx and S not in fx


def test_sdk_and_helper_names():
    assert S in _fx("import stripe\ndef f(c):\n    stripe.PaymentIntent.create(amount=5, customer=c)")
    assert _fx("import stripe\ndef f(c):\n    stripe.Refund.create(charge=c)") == {S}
    assert A in _fx("import boto3\ndef f():\n    boto3.client('iam').attach_group_policy("
                    "GroupName='g', PolicyArn='arn')")
    assert A in _fx("import stripe\ndef f(a):\n    stripe.Account.create_login_link(a)")
    assert S in _fx("def f(c, s, n):\n    _issue_supplier_refund(c, s, n)")
    assert R in _fx("def f(s3, b, k):\n    s3.copy_object(Bucket=b, CopySource=k, Key='a/' + k)")


def test_subprocess_arguments():
    assert A in _fx("import subprocess\ndef f(s, u):\n    subprocess.run(['/opt/bin/grant_access', "
                    "'--slot', s, '--user', u, '--role', 'attendee'])")
    assert S in _fx("import subprocess\ndef f(c):\n    subprocess.run(['billctl', 'refund', c])")
    # A billing *period* argument is not a payment.
    assert S not in _fx("import subprocess\ndef f(p):\n"
                        "    subprocess.run(['python3', 'export.py', '--period', p])")


def test_quiet_forms():
    # conn.close() is not a deletion; close_account is.
    assert D not in _fx("import sqlite3\ndef f():\n    c = sqlite3.connect('a')\n    c.close()")
    assert D in _fx("def f(a):\n    a.close_account()")
    # Constructors are not verbs (TransferResult(...) is not a transfer of money).
    assert S not in _fx("def f(a, b):\n    return TransferResult(source=a, destination=b)")


def test_locate_points_at_the_new_forms():
    code = ("import sqlite3\ndef f(i):\n    c = sqlite3.connect('a.db')\n"
            "    c.execute('SELECT 1 FROM orders WHERE id = ?', (i,))\n"
            "    c.execute('INSERT INTO refunds (order_id) VALUES (?)', (i,))\n    c.close()")
    assert locate(code, S) == [5]
    assert locate(code, D) == []
