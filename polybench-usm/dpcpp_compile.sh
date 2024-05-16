#!/bin/bash

PROG_LIST="2DConvolution 2mm 3DConvolution 3mm atax bicg correlation covariance gemm gesummv gramschmidt mvt syr2k syrk fdtd2d"

for prog in $PROG_LIST; do
  echo $prog
  clang++  -D__LLVM_SYCL__ -I../include -I../polybench/common   -fdiagnostics-color=always -fsycl -fsycl-targets=spir64 -O3 -DNDEBUG   -std=c++17 -o ${prog}-dpcpp ${prog}.cpp
done
