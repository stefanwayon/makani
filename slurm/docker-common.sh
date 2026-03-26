JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
# unique MPI port to avoid collisions when multiple jobs share a node
export MASTER_PORT=$((12355 + (JOB_ID % 100) * 100 + TASK_ID))
export DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)
CONTAINER_NAME="makani-${JOB_ID}_${TASK_ID}"
trap 'echo "TRAP: killing $CONTAINER_NAME"; docker kill $CONTAINER_NAME' SIGTERM SIGINT EXIT
