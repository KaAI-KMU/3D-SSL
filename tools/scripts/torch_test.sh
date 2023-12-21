#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch ${PY_ARGS}

