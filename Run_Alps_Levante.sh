#!/bin/bash
#=============================================================================
# Ported Alps ENUFFT/CSA sweep on DKRZ Levante
#=============================================================================
# Run from Levante with
#   cd enufft_paper_ready
#   sbatch Run_Alps_Levante.sh
#
# The script runs the exact dense CSA equations through Code_Alps.py.
# It parallelizes over independent (configuration, triangle) tasks with one
# Slurm rank per physical core and then deterministically reduces the chunks
# into the same final CSV products used by Plot_Alps.py.
#=============================================================================

#SBATCH --job-name=enufft_alps
#SBATCH -o slurm_enufft_alps-%j.out
#SBATCH -e slurm_enufft_alps-%j.err
#SBATCH --account=bb1097
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --exclusive

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

module load python3/2023.01-gcc-11.2.0

TASKS_PER_NODE="${TASKS_PER_NODE:-128}"
N_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
MAX_WORKER_RANKS="${MAX_WORKER_RANKS:-$(( N_NODES * TASKS_PER_NODE ))}"
distributed_prefix="./csv/Banerjee_2026_Enufft_Alps_Distributed_${SLURM_JOB_ID}"

if [ "${TASKS_PER_NODE}" -lt 1 ]; then
    echo "ERROR: TASKS_PER_NODE must be positive" >&2
    exit 1
fi

mkdir -p csv figures

python3 - <<'PY'
import numpy
import scipy
import matplotlib
import PIL
print("Python dependency check OK", flush=True)
PY

if [ ! -f ./srtm_alps/alps_dem_processed.npz ]; then
    echo "ERROR: Missing local DEM archive: ./srtm_alps/alps_dem_processed.npz" >&2
    echo "Build it first with:" >&2
    echo "  python3 Code_Alps_Preprocess.py" >&2
    exit 1
fi

python3 -u Code_Alps.py prepare

TOTAL_TASKS="$(cat "${distributed_prefix}_TaskCount.txt")"
WORKER_TASKS="${MAX_WORKER_RANKS}"
if [ "${TOTAL_TASKS}" -lt "${WORKER_TASKS}" ]; then
    WORKER_TASKS="${TOTAL_TASKS}"
fi
SRUN_TASKS_PER_NODE="${TASKS_PER_NODE}"
if [ "${WORKER_TASKS}" -gt 0 ] && [ "${WORKER_TASKS}" -lt "${SRUN_TASKS_PER_NODE}" ]; then
    SRUN_TASKS_PER_NODE="${WORKER_TASKS}"
fi

echo "=============================================================="
echo "Ported Alps ENUFFT/CSA sweep"
echo "=============================================================="
echo "Job: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "DEM: ./srtm_alps/alps_dem_processed.npz"
echo "Meshes: r2b4,r2b5"
echo "Distributed files: ${distributed_prefix}_*"
echo "Active triangle tasks: ${TOTAL_TASKS}"
echo "Worker ranks: ${WORKER_TASKS}"
echo "Tasks per node: ${SRUN_TASKS_PER_NODE}"
echo "Started: $(date)"
echo "=============================================================="

if [ "${WORKER_TASKS}" -gt 0 ]; then
    # Some ranks finish earlier because task costs differ. Wait indefinitely for
    # the slower ranks instead of letting srun terminate the whole step.
    srun --exact --wait=0 --ntasks="${WORKER_TASKS}" --ntasks-per-node="${SRUN_TASKS_PER_NODE}" --cpus-per-task=1 \
        --distribution=block:cyclic \
        python3 -u Code_Alps.py worker

    python3 -u Code_Alps.py reduce
else
    echo "No active triangle tasks; all requested configs already have summary CSVs."
fi

echo "=============================================================="
echo "Sweep complete: $(date)"
echo "CSV output: ./csv"
echo "=============================================================="
