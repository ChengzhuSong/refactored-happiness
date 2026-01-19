#!/usr/bin/env bash
# Template script to download large data/model artifacts from a cloud location.
# Edit URLs to point to your storage (S3/HTTP/Google Drive) and run this script.

set -euo pipefail

OUT_DIR="data/crello"
mkdir -p "$OUT_DIR"

# Example: download fused embeddings and a small sample model (replace with your links)
# curl -L -o "$OUT_DIR/crello_element_fused_embeddings.npy" "https://example.com/path/crello_element_fused_embeddings.npy"
# curl -L -o "models/masked_element_model.best.pt" "https://example.com/path/masked_element_model.best.pt"

echo "Download script template. Edit URLs and uncomment the curl lines to fetch assets."
