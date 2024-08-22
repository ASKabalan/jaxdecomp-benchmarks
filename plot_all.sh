#!/bin/bash

gpus="4 8 16 32 64"
sizes="1024 2048"
time_aggregations="mean"
time_columns="mean_time"
figure_size="15 7"
precisions=("float32" "float64")
scaling=("Weak" "Strong")
fft_types=("FFT" "IFFT")
dark_mode="True"
backend="NCCL"
jaxdecomp_a100_new="out/a100/JAXDECOMP.csv"
# jaxdecomp_a100_old="slabs_old/a100/old_JAXDECOMP.csv"
jaxdecomp_v100_new="out/v100/JAXDECOMP.csv"
# jaxdecomp_v100_old="slabs_old/v100/old_JAXDECOMP.csv"
jax_a100_new="out/a100/JAX.csv"
# jax_a100_old="slabs_old/a100/old_JAX.csv"
jax_v100_new="out/v100/JAX.csv"
# jax_v100_old="slabs_old/v100/old_JAX.csv"
time_columns="mean_time"
memory_columns="temp_size"

for size in ${sizes[@]}; do
  # mean time plotting
  for precision in ${precisions[@]}; do
      for fft_type in ${fft_types[@]}; do
        #Strong scaling
        # jhp plot -f $jaxdecomp_a100_new $jax_a100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type -db $dark_mode -o plots/gpus_a100_${fft_type}_${precision}_${size}_${time_columns}.png -sc Strong  -b $backend -pt $time_columns
        # jhp plot -f $jaxdecomp_a100_new $jax_a100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type -db $dark_mode -o plots/gpus_a100_${fft_type}_${precision}_${size}_used_mem.png -sc Strong  -b $backend -pm $memory_columns
        jhp plot -f $jaxdecomp_v100_new $jax_v100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type  -o plots/gpus_v100_${fft_type}_${precision}_${size}_${time_columns}.png -sc Strong  -b $backend -pt $time_columns
        jhp plot -f $jaxdecomp_v100_new $jax_v100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type  -o plots/gpus_v100_${fft_type}_${precision}_${size}_used_mem.png -sc Strong  -b $backend -pm $memory_columns
        #Weak scaling
        # jhp plot -f $jaxdecomp_a100_new $jax_a100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type -db $dark_mode -o plots/data_a100_${fft_type}_${precision}_${size}_${time_columns}.png -sc Weak  -b $backend -pt $time_columns
        # jhp plot -f $jaxdecomp_a100_new $jax_a100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type -db $dark_mode -o plots/data_a100_${fft_type}_${precision}_${size}_used_mem.png -sc Weak  -b $backend -pm $memory_columns
        jhp plot -f $jaxdecomp_v100_new $jax_v100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type  -o plots/data_v100_${fft_type}_${precision}_${size}_${time_columns}.png -sc Weak  -b $backend -pt $time_columns
        jhp plot -f $jaxdecomp_v100_new $jax_v100_new -g $gpus -d $size -ps plot_all -pr $precision -fs $figure_size -fn $fft_type  -o plots/data_v100_${fft_type}_${precision}_${size}_used_mem.png -sc Weak  -b $backend -pm $memory_columns
      done
  done
done
