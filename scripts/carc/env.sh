# Shared environment for CARC: sourced by setup.sh and train.slurm.
# The venv is built on the module's Python, so the same module must be loaded at run time.
# If `module load python` fails or gives < 3.10, run `module spider python` and set
# PYTHON_MODULE to a 3.10+ version, e.g. `export PYTHON_MODULE=python/3.11.9`.

module purge
module load "${PYTHON_MODULE:-python}"

export HF_HOME="/scratch1/$USER/hf"   # model downloads go to scratch, not the home quota
export PYTHONUNBUFFERED=1              # so the SLURM log shows each epoch as it finishes
