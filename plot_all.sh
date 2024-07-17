#!/bin/bash

gpus="4 8 16 32 64"
sizes="256 512 1024 2048 4096"
time_aggregations="min"
figure_size="10 5"
precisions=("float32" "float64")
scaling=("Weak" "Strong")
fft_types=("FFT" "IFFT")
dark_mode="False"


for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        python plotter.py -f out/v100/* -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -t $fft_type -d $sizes -db $dark_mode -o plots/gpus_v100_${fft_type}_${precision}.png -sc Strong
        python plotter.py -f out/v100/* -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -t $fft_type -d $sizes -db $dark_mode -o plots/data_v100_${fft_type}_${precision}.png -sc Weak
        python plotter.py -f out/a100/* -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -t $fft_type -d $sizes -db $dark_mode -o plots/gpus_a100_${fft_type}_${precision}.png -sc Strong
        python plotter.py -f out/a100/* -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -t $fft_type -d $sizes -db $dark_mode -o plots/data_a100_${fft_type}_${precision}.png -sc Weak
    done
done
