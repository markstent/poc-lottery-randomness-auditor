# poc-lottery-randomness-auditor

Reusable Python and AI proof-of-concept template.

## Technology

- Python 3.13
- uv
- pytest
- Ruff
- OpenCode
- LiteLLM
- 1Password
- Shared MCP gateway

## Getting Started

1. Create a repository using this GitHub template.
2. Clone the repository.
3. Rename the Python package and project metadata.
4. Run `uv sync`.
5. Create a project-specific LiteLLM virtual key.
6. Store the key in 1Password.
7. Copy `.ai-project.example` to `.ai-project`.
8. Set LITELLM_ITEM to the 1Password item name.
9. Start OpenCode using `oc`.

## Development

Install dependencies:

    uv sync

Run tests:

    uv run pytest

Run linting:

    uv run ruff check .

Check formatting:

    uv run ruff format --check .

Start the AI coding agent:

    oc

## Structure

- src/ - application code
- tests/ - automated tests
- docs/ - specifications and decisions
- experiments/ - exploratory work
- data/ - local datasets
- scripts/ - utilities

## Security

Never commit secrets or production data.

LiteLLM manages model access, MCP permissions,
project budgets and spend tracking.
