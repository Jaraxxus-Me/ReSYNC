#!/usr/bin/env bash
# Install only the optional demo environment; never modify .venv or .venv-icy.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d inference_assets/isaac/runtime/IsaacLab/source ]]; then
  echo 'Download the Isaac demo release assets before installation.' >&2
  exit 1
fi
uv venv --python 3.11 .venv-isaac
uv pip install --python .venv-isaac/bin/python --no-deps \
  --index-strategy unsafe-best-match -r release/isaac/requirements.lock
uv pip install --python .venv-isaac/bin/python --no-deps -e . \
  -e third-party/prpl-mono/prpl-utils \
  -e third-party/prpl-mono/prbench \
  -e third-party/prpl-mono/relational-structs \
  -e third-party/prpl-mono/toms-geoms-2d
for project in IsaacLab UWLab; do
  for package in inference_assets/isaac/runtime/"$project"/source/*; do
    case "$(basename "$package")" in
      isaaclab|isaaclab_assets|isaaclab_rl|isaaclab_tasks|uwlab|uwlab_assets|uwlab_rl|uwlab_tasks)
        uv pip install --python .venv-isaac/bin/python --no-deps -e "$package"
        ;;
    esac
  done
done
