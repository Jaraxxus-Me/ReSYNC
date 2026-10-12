#!/bin/bash
set -euo pipefail
# docformatter returns 3 when it successfully reformats a file in place.
docformatter -i -r . || { status=$?; [ "$status" -eq 3 ] || exit "$status"; }
isort .
python -m black .
