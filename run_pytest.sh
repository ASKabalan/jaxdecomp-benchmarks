#!/bin/bash

log_file=$(basename $1 .py)
python -m pytest -s -v &> ${log_file}_${SLURM_PROCID}.log