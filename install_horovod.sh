#!/usr/bin/env bash
# Run this AFTER `conda env create -f environment.yml` and
# `conda activate unbinned_inference`.
#
# horovod's setup.py checks these env vars at build time to decide
# which ops to compile. They must be set before pip runs -- setting
# them afterwards, or putting `horovod` in environment.yml's pip
# section, does nothing (that path builds CPU-only ops silently).
set -euo pipefail

# horovod's build (CMake) needs to import TensorFlow at configure time,
# to locate its headers/libs to link against. pip's default build
# isolation runs the build in a separate, temporary environment that
# does NOT have TensorFlow installed (only this env does), so the
# import fails there even though TF is present when you run python
# directly. --no-build-isolation fixes this by building in the active
# env instead -- but that means the build tools pip would normally
# supply in the isolated env have to be installed here manually first.
pip install cmake ninja pybind11 packaging setuptools wheel cffi

# horovod bundles a vendored copy of gloo whose CMakeLists.txt declares
# a very old cmake_minimum_required. CMake >=4.0 (which the line above
# installs) refuses to configure any project requesting policies older
# than 3.5. This env var tells CMake to treat those old declarations as
# 3.5 instead of erroring out -- it's CMake's own documented workaround
# for exactly this situation, not a horovod-specific hack.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# NCCL itself: tensorflow[and-cuda]'s pip-bundled NCCL (nvidia-nccl-cu12)
# ships only the runtime .so, no headers -- not enough for horovod's
# CMake build to link against. Point it at the cluster's NVIDIA HPC SDK
# install instead, which has both nccl.h and libnccl.so in the standard
# <HOME>/include, <HOME>/lib layout CMake's FindNCCL expects. Using the
# 12.8-matched NCCL 2.25 build (not the 11.8 copy) since that's the CUDA
# toolkit version CMake picks up on this system. If this path doesn't
# exist on a different machine, find it with:
#   find /opt/nvidia/hpc_sdk -iname nccl.h
HOROVOD_NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/nccl-2.25 \
HOROVOD_GPU_OPERATIONS=NCCL \
HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITHOUT_PYTORCH=1 \
HOROVOD_WITHOUT_MXNET=1 \
  pip install --no-cache-dir --no-build-isolation horovod

# Sanity check: confirms which frameworks/ops horovod actually built
# against. Look for "Horovod ... Available/Enabled: True" next to
# TensorFlow and NCCL specifically.
horovodrun --check-build