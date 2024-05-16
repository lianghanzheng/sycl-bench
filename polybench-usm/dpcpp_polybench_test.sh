#!/bin/bash

export POCL_AFFINITY=1

PROG_LIST=(2DConvolution 2mm 3DConvolution 3mm atax bicg correlation covariance gemm gesummv gramschmidt mvt syr2k syrk fdtd2d)

INPUT_LIST=(16384 1024 256 1024 16384 16384 1024 1024 1024 16384 512 8192 1024 1024 512)


for i in $(seq 0 13); do
  for THREAD_COUNT in 64 32 16 8 4 2 1; do
    export POCL_MAX_PTHREAD_COUNT=$THREAD_COUNT
    echo "-------------------------------------------------------"
    echo $PROG Thread-count=$THREAD_COUNT
    echo "-------------------------------------------------------"
    echo ${PROG_LIST[$i]} ${INPUT_LIST[$i]}
    ./${PROG_LIST[$i]}-dpcpp --size=${INPUT_LIST[$i]}
    #./${PROG_LIST[$i]}-dpcpp --size=256 --local=16
  done
done
