# Shared environment for CARC: sourced by setup.sh and train.slurm.
# The venv is built on the module's Python, so the same module must be loaded at run time.
# If `module load python` fails or gives < 3.10, run `module spider python` and set
# PYTHON_MODULE to a 3.10+ version, e.g. `export PYTHON_MODULE=python/3.11.9`.

module purge
module load "${PYTHON_MODULE:-python}"

export HF_HOME="/scratch1/$USER/hf"   # model downloads go to scratch, not the home quota
export PYTHONUNBUFFERED=1              # so the SLURM log shows each epoch as it finishes
# OpenBLAS starts one thread per core on import; login nodes cap threads per user, which made
# `import scipy` hang or segfault. Use the job's CPUs, or 1 on the login node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
