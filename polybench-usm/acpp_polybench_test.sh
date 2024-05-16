#!/bin/bash

export GOMP_CPU_AFFINITY="0-63"

PROG_LIST=(2DConvolution 2mm 3DConvolution 3mm atax bicg correlation covariance gemm gesummv gramschmidt mvt syr2k syrk fdtd2d)

INPUT_LIST=(16384 1024 256 1024 16384 16384 1024 1024 1024 16384 512 8192 1024 1024 512)


for i in $(seq 0 14); do
  for THREAD_COUNT in 64 32 16 8 4 2 1; do
    export OMP_NUM_THREADS=$THREAD_COUNT
    echo "-------------------------------------------------------"
    echo $PROG Thread-count=$THREAD_COUNT
    echo "-------------------------------------------------------"
    #echo ${PROG_LIST[$i]} ${INPUT_LIST[$i]}
    ./${PROG_LIST[$i]}-acpp --size=${INPUT_LIST[$i]}
  done
done
