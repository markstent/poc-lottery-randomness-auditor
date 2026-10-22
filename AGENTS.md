# POC Agent Instructions

## Principles
- Build professional, reproducible proof-of-concepts.
- Prefer simple, modular, maintainable code.
- Avoid over-engineering and unnecessary dependencies.
- Make the smallest correct change.
- Do not modify unrelated code.
- Be concise. Never use em dashes.

## Development
- Python 3.13 with uv.
- pytest for tests.
- Ruff for linting and formatting.
- Use type hints for public interfaces.
- Keep experiments separate from production code.
- Document significant technical decisions.

## Workflow
1. Read README.md and relevant documentation.
2. Inspect existing code before modifying it.
3. Propose a plan for substantial changes.
4. Implement in small, testable increments.
5. Run tests and lint checks.
6. Summarise changes, results and outstanding issues.

## Quality
- Write tests for important logic and edge cases.
- Prefer deterministic and reproducible results.
- Do not silently swallow exceptions.
- Avoid unnecessary abstractions.
- Use configuration rather than hard-coded values.

## AI and Models
- Route approved model calls through LiteLLM.
- Keep development and runtime model selection separate.
- Reuse globally configured MCP servers.
- Respect LiteLLM model permissions and budgets.
- Do not duplicate shared OpenCode configuration.

## Security
- Never commit secrets or API keys.
- Use 1Password for credentials.
- Never log or expose credentials.
- Use synthetic data unless explicitly authorised.
- Do not commit sensitive datasets.

## Human Approval
Ask before:
- Major architecture changes.
- Introducing significant dependencies.
- Destructive operations.
- Deployment or infrastructure changes.
- Paid external services.

## Project: lottery randomness auditor
- Purpose: audit the pinned UK Lotto draw snapshot for detectable deviations from fairness.
- Key commands: `uv run pytest`, `uv run ruff check .`, `bash scripts/verify_repro.sh`, and `uv run poc-lottery-randomness-auditor --snapshot data/snapshots/lotto --out reports`.
- Notebook: `notebooks/lotto_audit.py` (Marimo). It is linted with per-file ignores for reactive cell patterns.
- Design and plan: `docs/data-sources.md` and `docs/superpowers/plans/2026-10-21-lotto-randomness-auditor.md`.
- PDF output needs WeasyPrint system libraries (Pango/GObject); HTML output does not.
