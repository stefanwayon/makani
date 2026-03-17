#!/bin/bash
#SBATCH --job-name=convert-wb2
#SBATCH --gres=gpu:0
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

# Usage:
#   sbatch convert_wb2.sbatch --input_files <files...> --output_file <zarr>
#
# Examples:
#   sbatch convert_wb2.sbatch --input_files /data/runs/.../scores/rollouts_eu.h5 --output_file /data/runs/.../scores/rollouts_eu.zarr

NPROCS="${NPROCS:-4}"

cd ~/makani

set -x
DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) docker compose run --rm makani \
    mpirun -np ${NPROCS} --host localhost:${NPROCS} \
    python -u data_process/convert_makani_output_to_wb2.py \
      --verbose \
      "$@"
set +x