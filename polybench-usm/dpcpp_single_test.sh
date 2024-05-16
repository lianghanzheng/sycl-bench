#!/bin/bash

export POCL_AFFINITY=1

  for THREAD_COUNT in 64 32 16 8 4 2 1; do
    export POCL_MAX_PTHREAD_COUNT=$THREAD_COUNT
    echo "-------------------------------------------------------"
    echo $PROG Thread-count=$THREAD_COUNT
    echo "-------------------------------------------------------"
    #echo ${PROG_LIST[$i]} ${INPUT_LIST[$i]}
    ./$1-dpcpp --size=$2
  done
