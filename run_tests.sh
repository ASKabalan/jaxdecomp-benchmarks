#!/bin/bash

log_file=$(basename $1 .py)
python $@ &> ${log_file}_${SLURM_PROCID}.log
