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

FMI
- What does the raw data like:
  - `crello_test_elements_per_image.csv` — csv file with a row for an element, tokenized text, and image embeddings
- Some other scripts and how they work:
  - `encode_text_with_clip.py` — literal meanings, run 3 times for train. val and test set
    - actually not that useful
  - `prepare_poster_input.py` — prepare fixed-length input for transformer
    - For each element in a poster it builds a feature vector:
    - [image_embedding (512) | text_embedding (64) | geom (6)] → per-element feature dim (512+64+6 = 582 by default here).
    - It groups elements by poster, sorts elements within a poster, pads/truncates to a fixed slot count (max_elems = 64), and produces:
      - poster_inputs_X.npy (float32) shape: num_posters × max_elems × feat_dim
      - poster_inputs_mask.npy (uint8) shape: num_posters × max_elems (1 = occupied / valid)
      - poster_inputs_index.csv mapping poster id → number of elements (int)
    - The script is deliberately lightweight and can fall back to a deterministic text embedding if heavy HF deps aren't present.
  
