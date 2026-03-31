#!/bin/bash
#SBATCH --job-name=convert-wb2
#SBATCH --gres=gpu:0
#SBATCH --requeue
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
source ~/makani/slurm/docker-common.sh

docker compose run --name $CONTAINER_NAME --rm makani \
    mpirun -np ${NPROCS} --host localhost:${NPROCS} \
    python -u data_process/convert_makani_output_to_wb2.py \
      --verbose \
      "$@" &
wait $!
DOCKER_EXIT_CODE=$?
set +x