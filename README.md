# transform-crello

This repository contains code for a masked-element Transformer pipeline for poster element reconstruction. It includes scripts to compute CLIP embeddings, build fixed-length poster inputs, train a masked-element Transformer to reconstruct masked elements, produce visual reports, and run diagnostics.

Status
- Data and model artifacts (large files) are not included in the repo. See "Data & models" below for options to host or pull them.

Quick start (developer)
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a quick smoke training (small subset):

```bash
python3 scripts/train_masked_element_model.py --epochs 1 --subset 512 --batch 8
```

3. Generate a visual report (after you have model and data):

```bash
python3 scripts/generate_visual_reconstruction_report.py --test-x data/crello/poster_inputs_test_fused_X.npy --test-mask data/crello/poster_inputs_test_fused_mask.npy --use-fused --topk 3
```

Data & models
- This repo intentionally excludes large arrays and checkpoints. Options to obtain them:
  - Use Git LFS (suitable for a small number of large files). Example:
    - `git lfs install`
    - `git lfs track "*.npy"`
    - commit the generated `.gitattributes` and push.
  - Use DVC to manage large datasets and push them to an external remote (S3/GCS).
  - Host large assets in cloud storage (S3/GCS/HTTP) and provide a `scripts/download_data.sh` to fetch them.

Suggested workflow before pushing
1. Add this repo to GitHub (via `gh repo create` or the web UI).
2. Decide on a large-file strategy (Git LFS or DVC) and add relevant tracking.
3. Commit code and small metadata files; do not commit `data/` or `models/` unless tracked with LFS/DVC.

Developer notes
- Key scripts live under `scripts/`:
  - `train_masked_element_model.py` — train the masked-element Transformer
  - `train_element_attribute_encoder.py` — train per-element fusion encoder
  - `generate_poster_inputs_from_fused.py` — build fused poster inputs
  - `generate_visual_reconstruction_report.py` — HTML/CSV visual report
  - `compute_reconstruction_sensitivity.py` — ablation/sensitivity diagnostics
  - `eval_reconstruction_alignment.py` — evaluate cosine / P@k alignment

Contact
- If you need me to push a first commit for you (create README, .gitignore, initial commit), tell me and I can add those files here and/or run the git commands in your environment.
