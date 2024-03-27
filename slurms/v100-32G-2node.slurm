#!/bin/bash
##SBATCH --exclusive
#SBATCH --account=tkc@v100
#SBATCH --job-name=V100-2node     # Name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
#SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                  # decommenter pour reserver uniquement des GPU A100
# Ici, reservation de 8x10=80 CPU (4 taches par noeud) et de 8 GPU (4 GPU par noeud) sur 2 noeuds :
#SBATCH --nodes=2                 # nombre de noeud
#SBATCH --ntasks-per-node=4       # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4               # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
##SBATCH --gpus-per-task=2           # nombre de GPU par tache
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
# la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=V100-2node_decomp.out # name of output file
#SBATCH --error=V100-2node_decomp.out  # name of error file (here, in common with the output file)
# Cleans out modules loaded in interactive and inherited by default
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd



# Loading modules
module load python/3.10.4
source venv/v100/bin/activate
conda deactivate


module load nvidia-compilers/23.9 cuda/11.8.0 cudnn/8.9.7.29-cuda  openmpi/4.1.1-cuda nccl/2.18.5-1-cuda cmake

module load nvidia-nsight-systems/2022.5.1

export TMPDIR=$JOBSCRATCH
# Pour contourner un bogue dans les versions actuelles de Nsight Systems
# il est également nécessaire de créer un lien symbolique permettant de
# faire pointer le répertoire /tmp/nvidia vers TMPDIR
ln -s $JOBSCRATCH /tmp/nvidia

export OUTPUT_FOLDER_ARGS=1

function run_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: run_python <python_script> [arguments for the script]"
        return 1
    fi


    local script_name=$(basename "$1" .py)
    local output_dir="traces_2/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="traces_2/$script_name/$args"
    fi

    mkdir -p "$output_dir"

    srun python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}

# To Profile the script 
function profile_python() {
    if [ $# -lt 1 ]; then
        echo "Usage: profile_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="traces_2/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="traces_2/$script_name/$args"
    fi

    mkdir -p "$output_dir"

    srun nsys profile -t cuda,nvtx,osrt,mpi -o "$output_dir/report_rank%q{SLURM_PROCID}" python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}

function slaunch() {
    run_python "$@"
}

#export ENABLE_PERFO_STEP=NVTX


set -x
set +e
# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution with binding via bind_gpu.sh : 1 GPU per task

## Use l for local size (better because always divisible) or g for global size
## pfft3d -p <pdims> -l <size> -n <nodes> -b <backend> -o <output_folder>
## or 
## pfft3d -p <pdims> -g <size> -n <nodes> -b <backend> -o <output_folder>

# NCCL
slaunch benchmarks/pfft3d.py -p 2x4 -l 16 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 16 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 16 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 16 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 32 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 32 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 32 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 32 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 64 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 64 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 64 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 64 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 128 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 128 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 128 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 128 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 256 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 256 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 256 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 256 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 512 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 512 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 512 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 512 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 1024 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 1024 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 1024 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 1024 -b NCCL -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 2048 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 2048 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 2048 -b NCCL -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 2048 -b NCCL -n  2 -o  traces_2/pfft3d

# MPI
slaunch benchmarks/pfft3d.py -p 2x4 -l 16 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 16 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 16 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 16 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 32 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 32 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 32 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 32 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 64 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 64 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 64 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 64 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 128 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 128 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 128 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 128 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 256 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 256 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 256 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 256 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 512 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 512 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 512 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 512 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 1024 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 1024 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 1024 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 1024 -b MPI -n  2 -o  traces_2/pfft3d

slaunch benchmarks/pfft3d.py -p 2x4 -l 2048 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 4x2 -l 2048 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 1x8 -l 2048 -b MPI -n  2 -o  traces_2/pfft3d
slaunch benchmarks/pfft3d.py -p 8x1 -l 2048 -b MPI -n  2 -o  traces_2/pfft3d

# JAXFFT
slaunch benchmarks/jaxfft.py  -l 16 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 32 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 64 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 128 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 256 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 512 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 1024 -n  2 -o traces_2/jaxfft
slaunch benchmarks/jaxfft.py  -l 2048 -n  2 -o traces_2/jaxfft
