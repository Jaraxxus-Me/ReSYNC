#!/bin/bash
set -euo pipefail
./run_autoformat.sh
mypy .
pylint --rcfile=.pylintrc --recursive=y src scripts tests
# Run CPU tests; the remaining simulation suites require GPUs.
pytest tests/blocked_stacking tests/isaac_demo
