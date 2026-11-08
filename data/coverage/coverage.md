| Group | n | Agent Audit | Model @0.5 | Either | Both | Top Agent Audit rules |
|---|---|---|---|---|---|---|
| bug: eval_exec | 9 | 9 (100%) | not scored | - | - | AGENT-001 (9), AGENT-034 (9), AGENT-017 (9), AGENT-035 (8), AGENT-026 (4) |
| bug: hardcoded_secret | 9 | 9 (100%) | not scored | - | - | AGENT-022 (7), AGENT-004 (7), AGENT-026 (2), AGENT-111 (1), AGENT-034 (1) |
| bug: path_traversal | 9 | 8 (89%) | not scored | - | - | AGENT-026 (6), AGENT-034 (6), AGENT-022 (5) |
| bug: shell_injection | 9 | 9 (100%) | not scored | - | - | AGENT-034 (9), AGENT-022 (9), AGENT-035 (9), AGENT-001 (7), AGENT-026 (4) |
| bug: sql_injection | 9 | 9 (100%) | not scored | - | - | AGENT-034 (8), AGENT-041 (1) |
| bug: ssrf | 9 | 9 (100%) | not scored | - | - | AGENT-022 (9), AGENT-026 (9), AGENT-034 (3), AGENT-020 (1) |
| bug: unsafe_deserialization | 9 | 7 (78%) | not scored | - | - | AGENT-022 (5), AGENT-034 (5), AGENT-026 (4) |
| honest: declare_effect | 54 | 31 (57%) | 2 (4%) | 32 | 1 | AGENT-034 (17), AGENT-022 (9), AGENT-111 (9), AGENT-026 (8), AGENT-047 (2) |
| honest: honest_rewrite | 54 | 21 (39%) | 0 (0%) | 21 | 0 | AGENT-034 (9), AGENT-022 (9), AGENT-026 (6), AGENT-111 (3), AGENT-047 (2) |
| honest: original | 77 | 31 (40%) | 0 (0%) | 31 | 0 | AGENT-022 (16), AGENT-034 (14), AGENT-026 (8), AGENT-047 (4), AGENT-041 (3) |
| mismatch: inject | 54 | 31 (57%) | 52 (96%) | 53 | 30 | AGENT-034 (17), AGENT-022 (9), AGENT-111 (9), AGENT-026 (8), AGENT-047 (2) |
| mismatch: weaken | 12 | 7 (58%) | 7 (58%) | 9 | 5 | AGENT-022 (4), AGENT-034 (4), AGENT-026 (2), AGENT-111 (1) |
