#!/bin/bash
./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc
# Only run blocked_stacking tests, others require ManiSkill and GPUs
pytest tests/blocked_stacking
