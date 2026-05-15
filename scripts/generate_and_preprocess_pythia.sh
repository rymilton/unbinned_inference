#!/bin/bash
# generate_and_preprocess_pythia.sh
# Usage: ./generate_and_preprocess_pythia.sh <alpha_s> <n_events> <lepton_id>
# Example: ./generate_and_preprocess_pythia.sh 0.118 10000000 -11
set -e  # Exit on any error

# ── Arguments ────────────────────────────────────────────────────────────────
ALPHA_S=${1:-0.118}
N_EVENTS=${2:-100}
LEPTON_ID=${3:-"-11"}  # -11 is positron (default), 11 is electron

# Set lepton string for filename
if [ "${LEPTON_ID}" -eq -11 ]; then
    LEPTON_STR="eplus"
elif [ "${LEPTON_ID}" -eq 11 ]; then
    LEPTON_STR="eminus"
else
    echo "Error: lepton_id must be 11 or -11, got '${LEPTON_ID}'"
    exit 1
fi

# Convert event count into short string (e.g. 100k, 10m)
format_events() {
    local n=$1

    if (( n >= 1000000 )); then
        echo "$((n / 1000000))M"
    elif (( n >= 1000 )); then
        echo "$((n / 1000))K"
    else
        echo "${n}"
    fi
}

N_EVENTS_SHORT=$(format_events "${N_EVENTS}")

echo "================================================"
echo "Running pipeline with alpha_s=${ALPHA_S}, nEvents=${N_EVENTS}, lepton=${LEPTON_STR} (id=${LEPTON_ID})"
echo "================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PYTHIA_EXE="${REPO_DIR}/pythia_generation/pythia_H1"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_pythia_data.py"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_pythia.py"

PYTHIA_ROOT_DIR="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_files/"
PYTHIA_H5_DIR="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_h5/"

# Construct the input string to match the TString::Format filename in the C++ code:
# pythia_H1_alphaS<alpha_s>_<lepton>_<nevents>events
ALPHA_S_FMT=$(printf "%.4f" "$ALPHA_S")
INPUT_STRING="pythia_H1_alphaS${ALPHA_S_FMT}_${LEPTON_STR}_${N_EVENTS_SHORT}events"

echo ""
echo "── Step 1: Pythia generation ────────────────────────────────────────────"
echo "Executable : ${PYTHIA_EXE}"
echo "Output file: ${PYTHIA_ROOT_DIR}${INPUT_STRING}.root"
"${PYTHIA_EXE}" "${ALPHA_S}" "${N_EVENTS}" "${LEPTON_ID}"
echo "Pythia generation complete."

echo ""
echo "── Step 2: prepare_pythia_data.py ───────────────────────────────────────"
python "${PREPARE_SCRIPT}" \
    --input_directory  "${PYTHIA_ROOT_DIR}" \
    --output_directory "${PYTHIA_H5_DIR}" \
    --input_string     "${INPUT_STRING}" \
    --lepton_beam      "${LEPTON_ID}"
echo "Preparation complete."

echo ""
echo "── Step 3: preprocess_pythia.py ─────────────────────────────────────────"
python "${PREPROCESS_SCRIPT}" \
    --data_folder "${PYTHIA_H5_DIR}" \
    --file_name   "${INPUT_STRING}.h5"
echo "Preprocessing complete."

echo ""
echo "================================================"
echo "Pipeline finished successfully."
echo "Final output: ${PYTHIA_H5_DIR}${INPUT_STRING}_prep.h5"
echo "================================================"