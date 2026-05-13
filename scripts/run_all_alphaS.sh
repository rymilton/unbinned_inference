#!/bin/bash
# run_all_alphaS.sh
# Runs generate_and_preprocess_pythia.sh for alpha_s = 0.05, 0.10, ..., 0.50
# each with 1 million events.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/generate_and_preprocess_pythia.sh"

N_EVENTS=1000000
LEPTON_ID=-11

for i in $(seq 1 10); do
    ALPHA_S=$(awk "BEGIN {printf \"%.2f\", $i * 0.05}")
    echo "========================================"
    echo "Submitting alpha_s=${ALPHA_S}"
    echo "========================================"
    bash "${PIPELINE}" "${ALPHA_S}" "${N_EVENTS}" "${LEPTON_ID}"
done

echo "All jobs complete."