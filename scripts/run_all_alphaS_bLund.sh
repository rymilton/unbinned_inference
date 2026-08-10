#!/bin/bash
# run_all_alphaS_bLund.sh
# Runs generate_and_preprocess_pythia.sh over a grid of (alpha_s, b_lund) points,
# each with 1 million events, except alpha_s=0.118/b_lund=0.98 (the reference
# point), which uses 10 million.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/generate_and_preprocess_pythia.sh"

LEPTON_ID=-11

# alpha_s / b_lund / n_events grid points
GRID=(
    "0.100 0.2 1000000"
    "0.100 0.8 1000000"
    "0.100 1.4 1000000"
    "0.100 2.0 1000000"
    "0.125 0.2 1000000"
    "0.125 0.8 1000000"
    "0.125 1.4 1000000"
    "0.125 2.0 1000000"
    "0.150 0.2 1000000"
    "0.150 0.8 1000000"
    "0.150 1.4 1000000"
    "0.150 2.0 1000000"
    "0.175 0.2 1000000"
    "0.175 0.8 1000000"
    "0.175 1.4 1000000"
    "0.175 2.0 1000000"
    "0.200 0.2 1000000"
    "0.200 0.8 1000000"
    "0.200 1.4 1000000"
    "0.200 2.0 1000000"
    "0.13 1.7 1000000"
    "0.16 1.1 1000000"
    "0.19 0.5 1000000"
    "0.118 0.98 10000000"
)

for point in "${GRID[@]}"; do
    read -r ALPHA_S B_LUND N_EVENTS <<< "${point}"

    echo "========================================"
    echo "Submitting alpha_s=${ALPHA_S}, b_lund=${B_LUND}, n_events=${N_EVENTS}"
    echo "========================================"
    bash "${PIPELINE}" "${ALPHA_S}" "${N_EVENTS}" "${LEPTON_ID}" "${B_LUND}"
done

echo "All jobs complete."
