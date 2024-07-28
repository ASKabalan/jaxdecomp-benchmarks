#!/bin/bash

gpus="4 8 16 32 64"
sizes="1024"
time_aggregations="mean"
time_columns="mean_time"
figure_size="20 10"
precisions=("float32" "float64")
scaling=("Weak" "Strong")
fft_types=("FFT" "IFFT")
dark_mode="True"
backend="NCCL"
jaxdecomp_a100_new="slabs_new/a100/JAXDECOMP.csv"
jaxdecomp_a100_old="slabs_old/a100/old_JAXDECOMP.csv"
jaxdecomp_v100_new="slabs_new/v100/JAXDECOMP.csv"
jaxdecomp_v100_old="slabs_old/v100/old_JAXDECOMP.csv"
jax_a100_new="slabs_new/a100/JAX.csv"
jax_a100_old="slabs_old/a100/old_JAX.csv"
jax_v100_new="slabs_new/v100/JAX.csv"
jax_v100_old="slabs_old/v100/old_JAX.csv"



# compare new JAXDECOMP with new JAX
for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        hpc-plotter plot -f $jaxdecomp_a100_new $jax_a100_new -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -fn $fft_type -d $sizes -db $dark_mode -o plots/gpus_a100_NN_${fft_type}_${precision}.png -sc Strong -tc $time_columns -ps plot_all -b $backend
    done
done

# compare new JAXDECOMP with new JAX V100
for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        hpc-plotter plot -f $jaxdecomp_v100_new $jax_v100_new -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -fn $fft_type -d $sizes -db $dark_mode -o plots/gpus_v100_NN_${fft_type}_${precision}.png -sc Strong -tc $time_columns -ps plot_all -b $backend
    done
done

# compare old JAXDECOMP with old JAXDECOMP a100
for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        hpc-plotter plot -f $jaxdecomp_a100_old $jax_a100_old -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -fn $fft_type -d $sizes -db $dark_mode -o plots/gpus_a100_NO_${fft_type}_${precision}.png -sc Strong -tc $time_columns -ps plot_all -b $backend
    done
done

# compare old JAX with old JAX jax_a100
for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        hpc-plotter plot -f $jaxdecomp_v100_old $jax_v100_old -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -fn $fft_type -d $sizes -db $dark_mode -o plots/gpus_v100_JNO_${fft_type}_${precision}.png -sc Strong -tc $time_columns -ps plot_all -b $backend
    done
done

# compare old JAXDECOMP with new JAXDECOMP v100
for precision in ${precisions[@]}; do
    for fft_type in ${fft_types[@]}; do
        hpc-plotter plot -f $jaxdecomp_v100_old $jaxdecomp_v100_new -fs $figure_size -g $gpus -ta $time_aggregations -p $precision -fn $fft_type -d $sizes -db $dark_mode -o plots/gpus_v100_NO_${fft_type}_${precision}.png -sc Strong -tc $time_columns -ps plot_all -b $backend
    done
done
