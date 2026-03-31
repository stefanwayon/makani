JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
# unique MPI port to avoid collisions when multiple jobs share a node
export MASTER_PORT=$((12355 + (JOB_ID % 100) * 100 + TASK_ID))
export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)
CONTAINER_NAME="makani-${JOB_ID}_${TASK_ID}"
DOCKER_EXIT_CODE=0
MAX_RETRIES=${MAX_RETRIES:-3}

cleanup() {
    echo "TRAP: killing $CONTAINER_NAME"
    docker kill $CONTAINER_NAME 2>/dev/null
    if [[ $DOCKER_EXIT_CODE -ne 0 ]]; then
        ATTEMPT=${SLURM_RESTART_COUNT:-0}
        if [[ $ATTEMPT -lt $MAX_RETRIES ]]; then
            echo "Task failed (exit $DOCKER_EXIT_CODE), requeueing (attempt $((ATTEMPT + 1))/$MAX_RETRIES)"
            scontrol requeue ${JOB_ID}_${TASK_ID}
            scontrol update JobId=${JOB_ID}_${TASK_ID} Nice=1000
        else
            echo "Task failed (exit $DOCKER_EXIT_CODE), max retries ($MAX_RETRIES) reached"
        fi
    fi
    exit $DOCKER_EXIT_CODE
}
trap cleanup SIGTERM SIGINT EXIT
