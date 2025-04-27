#!/bin/bash
# Trap SIGTERM and execute a clean shutdown
trap 'echo "Caught SIGTERM signal, shutting down..."; exit 2' SIGTERM

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if MODEL_NAME and MODEL_DIR are provided
if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_NAME and MODEL_DIR must be provided." >&2
    exit 1
fi

# Build the command
CMD_ARGS="huggingface-cli download $MODEL_NAME --local-dir $MODEL_DIR --local-dir-use-symlinks False"

# Include token if provided
if [ -n "$HF_TOKEN" ]; then
    CMD_ARGS="$CMD_ARGS --token $HF_TOKEN"
fi

# Include exclude params if provided
if [ -n "$HF_EXTRA" ]; then
    CMD_ARGS="$CMD_ARGS $HF_EXTRA"
fi

if [ -n "$HF_HUB_ENABLE_HF_TRANSFER"]; then
    CMD_ARGS="HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER $CMD_ARGS";
else
    CMD_ARGS="HF_HUB_ENABLE_HF_TRANSFER=TRUE $CMD_ARGS";
fi

# Execute the command
rm -rf $MODEL_DIR
eval $CMD_ARGS

# Execute the command specified as CMD in the Dockerfile, or override command when creating a container
exec "$@"