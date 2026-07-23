#!/bin/bash
# =============================================================================
# setup_login_node.sh — OPTIONAL manual pre-installation helper
#
# You do NOT need to run this. The PBS job (submit_job.pbs) is fully
# self-contained and handles micromamba + env + deps installation itself.
#
# Run this ONLY if you want to pre-warm the environment before submitting
# (saves ~8 min on the first job) or to debug dependency issues manually.
#
# Usage (on login node, optional):
#   bash ~/planktonai/hpc/setup_login_node.sh
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "PlanktonAI — Optional login-node pre-warm"
echo "NOTE: The PBS job does all this automatically."
echo "      This script just pre-warms the environment."
echo "$(date)"
echo "============================================================"

MAMBA_BIN="${HOME}/.local/bin/micromamba"
MAMBA_ROOT="${HOME}/micromamba"
ENV_NAME="planktonai"
PYTHON="${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python3"
PIP="${MAMBA_ROOT}/envs/${ENV_NAME}/bin/pip"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"

# Step 1 — Install micromamba
echo ""
echo "--- micromamba ---"
if [ -x "${MAMBA_BIN}" ]; then
    echo "Already installed: $(${MAMBA_BIN} --version)"
else
    mkdir -p "${HOME}/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "${HOME}/.local/bin" --strip-components=1 bin/micromamba
    chmod +x "${MAMBA_BIN}"
    echo "Installed: $(${MAMBA_BIN} --version)"
fi

# Step 2 — Create Python 3.8 env
echo ""
echo "--- Python 3.8 environment ---"
if [ -f "${PYTHON}" ]; then
    echo "Already exists: $(${PYTHON} --version)"
else
    "${MAMBA_BIN}" create --name "${ENV_NAME}" --root-prefix "${MAMBA_ROOT}" \
        python=3.8 -c conda-forge --yes
fi

# Step 3 — Install requirements
echo ""
echo "--- Dependencies ---"
REQUIREMENTS="$(dirname "$(dirname "$(realpath "$0")")")/requirements.txt"
"${PIP}" install --upgrade pip --quiet
"${PIP}" install -r "${REQUIREMENTS}"
echo "Done. $(${PYTHON} --version)"

echo ""
echo "Pre-warm complete. Now just: qsub hpc/submit_job.pbs"
