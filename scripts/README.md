# To use on Jean Zay with the TKC project


```bash
cd $ALL_CCFRWORK/jaxDecomp
```

this will run everything

# Run bencmarks on V100-32G:

From the root of the jaxDecomp workspace

```bash
sbatch benchmarks/scripts/v100-32G.slurm
sbatch benchmarks/scripts/v100-32G-2nodes.slurm
sbatch benchmarks/scripts/v100-32G-3nodes.slurm
```

# to run only Single jobs interactively

```bash
salloc --account=tkc@v100 --nodes=1  --ntasks-per-node=4 --gres=gpu:4 -C a100 --hint=nomultithread --qos=qos_gpu-dev
```

then

```bash
# Define a Global shape using -g
srun python benchmarks/pfft3d.py -g 1024 -p 2x4 -b NCCL -o path_for_csv_folder
# Or a local shape using -l
srun python benchmarks/pfft3d.py -l 1024 -p 2x4 -b NCCL -o path_for_csv_folder
# To compare with other frameworks
srun python benchmarks/jaxfft.py -g 1024 -o path_for_csv_folder
srun python benchmarks/mpi4jafft.py -g 1024 -o path_for_csv_folder
# For more info
srun python benchmarks/pfft3d.py -h
```
