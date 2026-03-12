#!/bin/bash
#SBATCH --job-name=makani-inference
#SBATCH --gres=gpu:8
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

# Usage:
#   sbatch inference.sbatch [extra inference.py args...]
#
# Examples:
#   sbatch inference.sbatch --start_date 2025-01-01 --end_date 2025-12-31 --output_file rollouts.h5
#   sbatch inference.sbatch --output_channels u10m v10m t2m --output_region 60 36 350.7 45

NGPUS=${SLURM_GPUS_ON_NODE:-0}
ENS_PARALLEL=$((NGPUS / 4))

cd ~/makani

DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) docker compose run --rm makani \
    mpirun -np ${NGPUS} --host localhost:${NGPUS} \
    python -u makani/inference.py \
      --climatology_file /data/climatology.h5 \
      --mask_file /data/climatology_mask.h5 \
      --load_checkpoint flexible \
      --amp_mode bf16 \
      --ensemble_size 8 \
      --ensemble_parallel_size ${ENS_PARALLEL} \
      --h_parallel_size 1 \
      --w_parallel_size 4 \
      --date_step 6 \
      --wb2_compatible \
      --output_channels u10m v10m t2m u100m v100m \
          u1000 v1000 t1000 z1000 \
          u850  v850  t850  z850 \
          u500  v500  t500  z500 \
      --output_region 60 36 350.7 45 \
      --output_memory_buffer_size 10 \
      "$@"

