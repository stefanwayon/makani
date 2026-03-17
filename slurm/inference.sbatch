#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --gres=gpu:8
#SBATCH --output=logs/slurm-%A_%a.log
#SBATCH --error=logs/slurm-%A_%a.log

# Usage:
#   sbatch inference.sbatch [extra inference.py args...]
#   sbatch --array=0-11 inference.sbatch [extra inference.py args...]
#
# Examples:
#   sbatch inference.sbatch --start_date 2025-01-01 --end_date 2025-12-31 --output_file rollouts.h5

NGPUS=${SLURM_GPUS_ON_NODE:-0}
ENS_PARALLEL=$((NGPUS > 0 ? NGPUS / 4 : 1))

# Parse args we need to intercept; everything else goes into ARGS
ARGS=()
START_DATE="2025-01-01"
END_DATE="2025-12-31"
OUTPUT_FILE=""
METRICS_FILE="metrics.h5"
BIAS_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_date)   START_DATE="$2"; shift 2 ;;
        --end_date)     END_DATE="$2"; shift 2 ;;
        --output_file)  OUTPUT_FILE="$2"; shift 2 ;;
        --metrics_file) METRICS_FILE="$2"; shift 2 ;;
        --bias_file)    BIAS_FILE="$2"; shift 2 ;;
        *)              ARGS+=("$1"); shift ;;
    esac
done

# Array mode: split date range and prefix output file
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    SHARD_FMT=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

    # Split date range evenly across tasks
    START_EPOCH=$(date -d "$START_DATE" +%s)
    END_EPOCH=$(date -d "$END_DATE" +%s)
    TOTAL=$((END_EPOCH - START_EPOCH))
    CHUNK=$((TOTAL / SLURM_ARRAY_TASK_COUNT))
    CHUNK_START=$((START_EPOCH + SLURM_ARRAY_TASK_ID * CHUNK))
    CHUNK_END=$((START_EPOCH + (SLURM_ARRAY_TASK_ID + 1) * CHUNK))
    # Last task gets the original end date; others subtract 1 day to avoid overlap (dates are inclusive)
    if [[ $SLURM_ARRAY_TASK_ID -eq $((SLURM_ARRAY_TASK_COUNT - 1)) ]]; then
        CHUNK_END=$END_EPOCH
    else
        CHUNK_END=$((CHUNK_END - 86400))
    fi
    START_DATE=$(date -d "@$CHUNK_START" +%Y-%m-%d)
    END_DATE=$(date -d "@$CHUNK_END" +%Y-%m-%d)

    # Prefix output files
    if [[ -n "$OUTPUT_FILE" ]]; then
        OUTPUT_FILE="${SHARD_FMT}__${OUTPUT_FILE}"
    fi
    if [[ -n "$METRICS_FILE" ]]; then
        METRICS_FILE="${SHARD_FMT}__${METRICS_FILE}"
    fi
    if [[ -n "$BIAS_FILE" ]]; then
        BIAS_FILE="${SHARD_FMT}__${BIAS_FILE}"
    fi
fi

# Rebuild the intercepted args
[[ -n "$START_DATE" ]]   && ARGS+=(--start_date "$START_DATE")
[[ -n "$END_DATE" ]]     && ARGS+=(--end_date "$END_DATE")
[[ -n "$OUTPUT_FILE" ]]  && ARGS+=(--output_file "$OUTPUT_FILE")
[[ -n "$METRICS_FILE" ]] && ARGS+=(--metrics_file "$METRICS_FILE")
[[ -n "$BIAS_FILE" ]]    && ARGS+=(--bias_file "$BIAS_FILE")

# Run inference
cd ~/makani

set -x
export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)
exec docker compose run --rm makani \
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
      "${ARGS[@]}"
set +x
