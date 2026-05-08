#!/bin/bash

input_file_name="dummy_input.h5"
output_file_name="dummy_output.h5"
max_num_jets=5
jet_radius=1.0
fastjet_config_path="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/bin/fastjet-config"
centauro_library_path="/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/lib"
while true; do
if [ "$1" = "--input" ]; then
   input_file_name=$2
   shift 2 # past argument
elif [ "$1" = "--output" ]; then
   output_file_name=$2
   shift 2 # past argument
elif [ "$1" = "--max_num_jets" ]; then
   max_num_jets=$2
   shift 2 # past argument
elif [ "$1" = "--jet_radius" ]; then
   jet_radius=$2
   shift 2 # past argument
elif [ "$1" = "--fastjet_config_path" ]; then
   fastjet_config_path=$2
   shift 2 # past argument
elif [ "$1" = "--centauro_library_path" ]; then
   centauro_library_path=$2
   shift 2 # past argument
else
   break
fi
done
g++ cluster_centauro.cxx -o cluster_centauro \
    `${fastjet_config_path} --cxxflags --libs --plugins` \
    `root-config --cflags --glibs` \
    -L${centauro_library_path} -lCentauro
./cluster_centauro --input "${input_file_name}" --output "${output_file_name}" --jet_radius "${jet_radius}" --max_num_jets "${max_num_jets}"