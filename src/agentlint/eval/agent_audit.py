"""Run Agent Audit (v0.19.2) on the same test set and compare coverage.

Gotchas:
- Agent Audit suppresses findings under test-looking paths (/tests/, /fixtures/, /examples/,
  /samples/, /mocks/ ...). Write cases under neutral paths, e.g.
  probes/aa_run/<case>/order_service/tools/orders.py.
- Always include a positive control (eval / shell=True). A silent run without a firing
  control proves nothing.
"""
