# jaxDecomp-benchmarks

Benchmarks done on Jean-Zay using 1 ,2 ,3 and 4 nodes on v100 and a100 GPUs

out folder contains csv files of all runs

# Plotting

To Plot we use one CSV per framework (so all GPU runs together) the files are [benchmarks64/JAXDECOMP.csv](benchmarks64/JAXDECOMP.csv), [benchmarks64/JAX.csv](benchmarks64/JAX.csv), [benchmarks32/JAXDECOMP.csv](benchmarks32/JAXDECOMP.csv), [benchmarks32/JAX.csv](benchmarks32/JAX.csv),

Then use `python plotter.py -h` for more info

Example to plot data with Weak/Strong scaling

```bash
python plotter.py -f out/v100/* -fs 12 8 -g 4 8 16 32  -ta min -p float64 -t FFT -d 512 1024 2048 4096 -db False -o plots/gpus_v100_FFT_64.png -sc Strong
python plotter.py -f out/v100/* -fs 12 8 -g 4 8 16 32  -ta min -p float64 -t FFT -d 512 1024 2048 4096 -db False -o plots/gpus_v100_FFT_64.png -sc Weak
```

to learn more about the options use `python plotter.py -h`


# Plot everything

Use script `plot_all.sh` to plot all the data


# Important params

- `-ta` : time aggregation : min, max, mean : how to aggregate the time accross the ranks

since there is a `block_until_ready` in the code, the min is the most relevant 

- `-sc` : Strong or Weak scaling


# Running slurms

Slurms are made for Jean-Zay, you can use them as a template for your own cluster

