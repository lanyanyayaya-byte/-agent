"""Centralized execution budgets and retry policy for plain chat and Skills Runs.

Keep all call counts, timeouts, recursion limits, and retry parameters here so they
can be inspected and tuned from a single location without touching unrelated modules.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plain / Legacy Chat (native file grants & MCP without Skills) — UNCHANGED
# ---------------------------------------------------------------------------
LEGACY_MAX_TOOL_CALLS: int = 8
LEGACY_MAX_RUN_SECONDS: float = 300.0
LEGACY_MAX_TOOL_CONTEXT_CHARS: int = 64_000
LEGACY_MAX_TOOL_CONTEXT_BYTES: int = 96_000

# ---------------------------------------------------------------------------
# Skills / Workspace / Runtime Runs (P9.7 relaxed budgets)
# ---------------------------------------------------------------------------
SKILLS_MAX_TOTAL_TOOL_CALLS: int = 100
SKILLS_MAX_CATEGORY_TOOL_CALLS: int = 80
SKILLS_MAX_LEGACY_TOOL_CALLS: int = 20
SKILLS_RECURSION_LIMIT: int = 250
SKILLS_MAX_RUN_SECONDS: float = 1800.0
SKILLS_MAX_TOOL_CONTEXT_CHARS: int = 400_000
SKILLS_MAX_TOOL_CONTEXT_BYTES: int = 600_000

# Anti-loop guard: identical consecutive recoverable tool errors still stop at 3.
MAX_CONSECUTIVE_REPEAT_FAILURES: int = 3

# ---------------------------------------------------------------------------
# Model Request Retry & Output-Length Recovery Policy
# ---------------------------------------------------------------------------
MODEL_RETRY_MAX_ATTEMPTS: int = 3
MODEL_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.5, 3.0, 6.0)
SKILLS_LENGTH_RECOVERY_MAX_ATTEMPTS: int = 2
