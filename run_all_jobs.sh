#!/bin/bash
# Run all slurms jobs
nodes_v100=(1 2 4 8 16 32)
nodes_a100=(1 2 4 8 16 32)


for n in ${nodes_v100[@]}; do
    sbatch --account=tkc@v100 --nodes=$n --gres=gpu:4 --tasks-per-node=4 -C v100-32g --job-name=FFT-$n-N-v100 fft_benchmarking.slurm
done

for n in ${nodes_a100[@]}; do
    sbatch --account=tkc@a100 --nodes=$n --gres=gpu:8 --tasks-per-node=8 -C a100     --job-name=FFT-$n-N-a100 fft_benchmarking.slurm
done
