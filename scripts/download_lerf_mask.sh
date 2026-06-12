#!/usr/bin/env bash
# Download LERF-Mask from HuggingFace (Gaussian-Grouping).
# https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/datasets/lerf_mask"
TMP="${ROOT}/datasets/_gaussian_grouping_hf"

echo "Downloading LERF-Mask to ${DEST} ..."

if ! command -v huggingface-cli &>/dev/null; then
  echo "Install huggingface_hub: pip install huggingface_hub"
  exit 1
fi

mkdir -p "${ROOT}/datasets"
huggingface-cli download mqye/Gaussian-Grouping \
  --include "data/lerf_mask/*" \
  --local-dir "${TMP}" \
  --local-dir-use-symlinks False

SRC="${TMP}/data/lerf_mask"
if [[ ! -d "${SRC}" ]]; then
  echo "Expected ${SRC} after download."
  exit 1
fi

rm -rf "${DEST}"
mv "${SRC}" "${DEST}"
rm -rf "${TMP}"

echo "Done. Scenes: figurines, ramen, teatime under ${DEST}"
