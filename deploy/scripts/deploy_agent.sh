#!/usr/bin/env bash
# Deploy the agent to AgentCore Runtime.
#
# `deploy/runtime/` is a build directory, not a second copy of the project: everything
# but the entrypoint and the dependency list is rebuilt from source here, every
# time, so it cannot be deployed stale. Whole packages are copied rather than
# named files, so a module added to one of them needs no change here.
set -euo pipefail
cd "$(dirname "$0")/../.."

rm -rf deploy/runtime/agent deploy/runtime/demo deploy/runtime/your-config
cp -r agent your-config deploy/runtime/
# The bundle mirrors the source layout, so the imports inside it resolve the
# same way they do here. Only the parts of demo/ the runtime imports come
# along - the page, the deck and the tests have no business in a microVM.
mkdir -p deploy/runtime/demo
cp demo/tools.py deploy/runtime/demo/
cp -r demo/simulator demo/evals demo/lab deploy/runtime/demo/
# The command line belongs to a laptop and drags in the terminal dossier.
rm -f deploy/runtime/demo/lab/cli.py
find deploy/runtime -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import pathlib
import sys

# What the entrypoint imports has to be what was copied, or the first request
# of the day is the one that finds out.
missing = [
    name for name in ("agent", "your-config", "demo/simulator", "demo/evals",
                      "demo/lab")
    if not (pathlib.Path("deploy/runtime") / name).is_dir()
]
if missing:
    sys.exit(f"deploy/runtime/ is missing {', '.join(missing)}")
for module in ("demo/lab/stream.py", "demo/lab/process.py", "demo/lab/run.py",
               "demo/tools.py", "agent/events.py"):
    if not (pathlib.Path("deploy/runtime") / module).is_file():
        sys.exit(f"deploy/runtime/ is missing {module}")

# Compiled, not parsed. A syntax error that only the compiler catches - `yield
# from` inside an async generator is one - stops the module importing, and a
# runtime whose entrypoint will not import never finishes starting.
for source in sorted(pathlib.Path("deploy/runtime").rglob("*.py")):
    try:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    except SyntaxError as problem:
        sys.exit(f"{source}: {problem}")
print("  deploy/runtime/ rebuilt from source and compiles")
PY

# The role's policy is kept with the two things that are nobody else's -
# the account and the bucket - written as names rather than values, and
# filled in here. What is committed can be read by anyone; what is deployed
# is this account's.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET_NAME="${LABCAMP_BUCKET:?set LABCAMP_BUCKET to the data bucket of the stack}"
sed -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" -e "s/BUCKET_NAME/${BUCKET_NAME}/g" \
    deploy/runtime/policies/labcamp.json > deploy/runtime/policies/.this-account.json
echo "  policy rendered for ${ACCOUNT_ID}"

# The CLI finds its project in the working directory, and `agentcore.json`
# names `runtime/` relative to the same place - so both resolve from `deploy/`.
cd deploy
exec agentcore deploy -y "$@"
