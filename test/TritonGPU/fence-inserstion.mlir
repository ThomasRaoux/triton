// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s

