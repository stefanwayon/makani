#!/bin/bash
#SBATCH --job-name=training
#SBATCH --gres=gpu:8
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

# Usage:
#   sbatch training.sbatch [extra ensemble.py args...]
#
# Examples:
#   sbatch training.sbatch --yaml_config ... --config fcn3_ensemble_2lw --run_num ifs_finetune
set -x

DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) docker compose run --rm makani \
    mpirun -np 8 --host localhost:8 \
    python -u makani/ensemble.py \
      --load_checkpoint flexible \
      --amp_mode bf16 \
      --h_parallel_size 2 \
      --w_parallel_size 4 \
      --multistep_count 8 \
      --checkpointing_level 3 \
      "$@"