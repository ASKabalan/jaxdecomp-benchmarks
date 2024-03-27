# jaxDecomp-benchmarks

Benchmarks done on Jean-Zay using 1 , 2 and 4 nodes with 4 , 8 and 16 GPUs respectively

Each benchmark folder has three folders

 - one node : trace
 - two node : trace_2
 - four node : trace_4

The folder name indicates the parameters used in the slurm\
for example `benchmarks32/traces_4/pfft3d/p_1x16_l_64_b_MPI_n_4_o_traces_4pfft3d`

This was done using single precision floats, on 4 nodes, using 1x16 slab decomposition, using MPI as backend and with 64^3 per device (local size) so the global size is 64 * 16 = (1024 , 1024 , 1024)

# CSV files

Each run has a CSV file for example `benchmarks32/traces_4/pfft3d/jaxdecompfft.csv`\
Has the benchmark result for all srun of that particular run

# Outputs

Outputs show if the job failed, and they also show that the FFT give the right result (Maximum reconstruction difference is ~0)\
For example [benchmarks32/traces_4/pfft3d/p_2x8_l_128_b_NCCL_n_4_o_traces_4pfft3d/pfft3d.out](benchmarks32/traces_4/pfft3d/p_2x8_l_128_b_NCCL_n_4_o_traces_4pfft3d/pfft3d.out)

Example of memory allocation faliure by JAX [benchmarks32/traces_4/jaxfft/l_1024_n_4_o_traces_4jaxfft/jaxfft.err](benchmarks32/traces_4/jaxfft/l_1024_n_4_o_traces_4jaxfft/jaxfft.err)


# Plotting

To Plot we use one CSV per framework (so all GPU runs together) the files are [benchmarks64/JAXDECOMP.csv](benchmarks64/JAXDECOMP.csv), [benchmarks64/JAX.csv](benchmarks64/JAX.csv), [benchmarks32/JAXDECOMP.csv](benchmarks32/JAXDECOMP.csv), [benchmarks32/JAX.csv](benchmarks32/JAX.csv),

Then use `python plotter.py -h` for more info

Example to plot data as X axis with fixed GPUs

```bash
python plotter.py -f benchmarks32/JAX.csv benchmarks32/JAXDECOMP.csv -g 4 8 16 32 -s JAX -fs 12 8 -nl -o plots/single_precision_gpus.png
python plotter.py -f benchmarks64/JAX.csv benchmarks64/JAXDECOMP.csv -g 4 8 16 32 -s JAX -fs 12 8 -nl -o plots/double_precision_gpus.png
```

To plot GPUs as X axis with fixed data

```bash
python plotter.py -f benchmarks32/JAX.csv benchmarks32/JAXDECOMP.csv -d 128 256 512 1024 2048 -s JAX -fs 14 12 -nl -o plots/single_precision_data.png
python plotter.py -f benchmarks64/JAX.csv benchmarks64/JAXDECOMP.csv -d 128 256 512 1024 2048 -s JAX -fs 14 12 -nl -o plots/double_precision_data.png
```

My plots are in `plots` folder
