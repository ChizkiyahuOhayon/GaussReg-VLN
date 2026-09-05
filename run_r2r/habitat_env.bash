#!/bin/bash

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate the HAVLN-CE Conda environment first." >&2
    return 1
fi

for candidate in \
    "${CONDA_PREFIX}"/lib/python*/site-packages/habitat_sim-*.egg/habitat_sim/_ext; do
    if [ -f "${candidate}/libCorradePluginManager.so.2" ]; then
        HABITAT_SIM_EXT=${candidate}
        break
    fi
done

if [ ! -f "${HABITAT_SIM_EXT:-}/libCorradePluginManager.so.2" ]; then
    echo "Cannot find Habitat-Sim native libraries under ${CONDA_PREFIX}." >&2
    return 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HABITAT_LAB=${HABITAT_LAB:-"$(dirname "${REPO_ROOT}")/habitat-lab"}
export HABITAT_SIM_EXT
export PYTHONPATH="${HABITAT_LAB}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${HABITAT_SIM_EXT}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
