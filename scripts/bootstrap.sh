#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${1:?Usage: scripts/bootstrap.sh <project-name>}"

# Validate project name
if [[ ! "$PROJECT_NAME" =~ ^[a-z][a-z0-9-]*$ ]]; then
  echo "Error: use lowercase letters, numbers and hyphens."
  exit 1
fi

PACKAGE_NAME="${PROJECT_NAME//-/_}"

# Prevent accidental reinitialisation
if [[ ! -d src/lc_poc_template ]]; then
  echo "Error: template package not found. Already bootstrapped?"
  exit 1
fi

# Rename Python package
mv src/lc_poc_template "src/$PACKAGE_NAME"

# Update project metadata
sed -i '' "s/lc-poc-template/$PROJECT_NAME/g" pyproject.toml
sed -i '' "s/lc_poc_template/$PACKAGE_NAME/g" pyproject.toml

# Update project title
sed -i '' "1s/.*/# $PROJECT_NAME/" README.md

# Create local project identity
cat > .ai-project <<CONFIG
LITELLM_ITEM=${PROJECT_NAME}-LiteLLM-dev
CONFIG

# Update dependencies and lockfile
uv lock
uv sync

# Validate
uv run pytest
uv run ruff check .
uv run ruff format --check .

echo ""
echo "Project initialised: $PROJECT_NAME"
echo "Next:"
echo "1. Create the LiteLLM virtual key: ${PROJECT_NAME}-dev"
echo "2. Store it in 1Password as ${PROJECT_NAME}-LiteLLM-dev"
echo "3. Start OpenCode using: oc"
