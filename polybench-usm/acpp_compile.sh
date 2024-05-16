#!/bin/bash

PROG_LIST="2DConvolution 2mm 3DConvolution 3mm atax bicg correlation covariance gemm gesummv gramschmidt mvt syr2k syrk fdtd2d"

for prog in $PROG_LIST; do
  echo $prog
  acpp -I../include -I../polybench/common -fdiagnostics-color=always -O3 -DNDEBUG -std=c++17 -o ${prog}-acpp ${prog}.cpp
done
