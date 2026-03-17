#!/bin/bash
set -e

# Wrapper that submits inference + convert pipeline.
#
# Usage:
#   slurm/run_inference.sh [sbatch args...] -- [inference.py args...]
#
# Examples:
#   slurm/run_inference.sh -- --config fcn3_ensemble_2lw --output_file rollouts_eu.h5
#   slurm/run_inference.sh --array=0-11 -- --config fcn3_ensemble_2lw --output_file rollouts_eu.h5

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Split args on "--": sbatch args before, inference args after
SBATCH_ARGS=()
INFERENCE_ARGS=()
past_sep=false
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        past_sep=true
    elif $past_sep; then
        INFERENCE_ARGS+=("$arg")
    else
        SBATCH_ARGS+=("$arg")
    fi
done

# Scan inference args for values we need for the convert step
CONFIG=""
RUN_NUM="00"
OUTPUT_FILE=""
for i in "${!INFERENCE_ARGS[@]}"; do
    case "${INFERENCE_ARGS[$i]}" in
        --config)      CONFIG="${INFERENCE_ARGS[$((i+1))]}" ;;
        --run_num)     RUN_NUM="${INFERENCE_ARGS[$((i+1))]}" ;;
        --output_file) OUTPUT_FILE="${INFERENCE_ARGS[$((i+1))]}" ;;
    esac
done

# Check if --array was passed to determine task count
N_TASKS=1
for arg in "${SBATCH_ARGS[@]}"; do
    if [[ "$arg" =~ ^--array=([0-9]+)-([0-9]+) ]]; then
        N_TASKS=$(( ${BASH_REMATCH[2]} - ${BASH_REMATCH[1]} + 1 ))
    fi
done

# Submit inference job
JOB=$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/inference.sbatch" "${INFERENCE_ARGS[@]}")
echo "Submitted inference job: $JOB ($N_TASKS task(s))"

# Submit convert job if output file is set
if [[ -n "$OUTPUT_FILE" && -n "$CONFIG" ]]; then
    SCORES_DIR=/data/runs/${CONFIG}/${RUN_NUM}/scores

    H5_FILES=()
    if [[ $N_TASKS -gt 1 ]]; then
        for i in $(seq 0 $((N_TASKS - 1))); do
            H5_FILES+=("${SCORES_DIR}/$(printf '%02d' $i)__${OUTPUT_FILE}")
        done
    else
        H5_FILES+=("${SCORES_DIR}/${OUTPUT_FILE}")
    fi
    ZARR_FILE="${SCORES_DIR}/${OUTPUT_FILE%.h5}.zarr"

    CONVERT_JOB=$(sbatch --parsable --dependency=afterok:${JOB} --gres=gpu:0 \
        "${SCRIPT_DIR}/convert_rollouts.sbatch" \
        --input_files ${H5_FILES[@]} \
        --output_file ${ZARR_FILE})
    echo "Submitted convert job: $CONVERT_JOB (depends on $JOB)"
fi
