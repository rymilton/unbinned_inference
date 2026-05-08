#!/bin/bash

batch_size=1000000
num_events=10000000
input_file="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/pythia_files/pythia_H1_alphaS14_eplus_10mil.root"
output_string="/global/cfs/cdirs/m3246/rmilton/unbinned_inference/clustered_pythia_files/pythia_H1_alphaS14_eplus_10mil_jets"
list_of_outputs=()

lepton_beam=-11
for i in {1..10}
do
    echo "Running iteration $i"
    output_file="${output_string}_${i}.root"
    list_of_outputs+=($output_file)
    python cluster_jets.py --input $input_file --output $output_file --start $(( (i-1) * batch_size )) --end $(( i * batch_size )) --use_breit --use_centauro --lepton_beam $lepton_beam
done
# Merge the output files
hadd -f "${output_string}_merged.root" "${list_of_outputs[@]}"