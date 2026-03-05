# MIDDAY Full Context Drop

## 0) Snapshot
- Timestamp: 2026-03-04 (America/Chicago), midday handoff
- Workspace: `/Users/josh/Desktop/emo_classifier`
- Git branch: `main`
- HEAD: `c0e7985`
- Remote: `git@github.com:jram4/emo_classifier.git`
- Status:
  - Synced with `origin/main`
  - Uncommitted files:
    - `scripts/build_relabel_queue.py`
    - `scripts/download_jamendo_subset.py`

---

## 1) What Was Built
The repo is now a functional Python emo-song classifier pipeline (audio in -> emo probability out), centered on CLAP embeddings + lightweight supervised model.

Core stack implemented:
- `CLAP` embeddings (`laion/clap-htsat-fused`)
- Logistic regression classifier
- CLI:
  - `emo-classifier train`
  - `emo-classifier predict`
  - `emo-classifier zero-shot`

Main code shipped in commit `c0e7985`:
- `src/emo_classifier/audio.py`
- `src/emo_classifier/embeddings.py`
- `src/emo_classifier/training.py`
- `src/emo_classifier/inference.py`
- `src/emo_classifier/cli.py`
- `scripts/build_labels_from_fma.py`
- `scripts/build_labels_from_jamendo.py`
- `scripts/build_labels_from_spotify.py`
- `scripts/export_spotify_playlist_tracks.py`
- `scripts/merge_label_csvs.py`
- `scripts/build_labels_from_folders.py`
- `scripts/run_fma_first_pass.sh`
- `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`

Compatibility and robustness improvements included:
- Transformers v5-safe embedding extraction path handling
- Offline fallback behavior (`--local-files-only`)
- Local snapshot resolution for CLAP cache
- `--skip-errors` path for dirty/partial datasets

---

## 2) Local Data State (Important)

### FMA (present locally)
- `external/fma/fma_small.zip` (~7.2 GB)
- `external/fma/fma_metadata.zip` (~342 MB)
- Extracted audio root:
  - `/Users/josh/Desktop/emo_classifier/external/fma/fma_small/fma_small`
- Extracted MP3 count: `8000`

### Jamendo metadata repo (present locally)
- `external/mtg-jamendo-dataset` exists (metadata/code repo cloned locally)
- Contains annotation TSVs and scripts
- No large downloaded Jamendo MP3 corpus confirmed in this repo snapshot

### Labels generated locally
- `/Users/josh/Desktop/emo_classifier/data/labels_fma_train.csv`
  - total `2519` rows
  - `emo=519`, `non_emo=2000`
- `/Users/josh/Desktop/emo_classifier/data/labels_fma_train_1000.csv`
  - total `1000` rows
  - `emo=500`, `non_emo=500`
- `/Users/josh/Desktop/emo_classifier/data/labels_fma_train_200.csv`
  - total `200` rows
  - `emo=100`, `non_emo=100`
- Also present:
  - `data/labels_fma_candidates.csv`
  - `data/labels_jamendo_candidates.csv`
  - `data/labels_train_candidates.csv`
  - `data/spotify_labels.csv` (tiny placeholder)

Note: most generated data/artifacts are intentionally gitignored.

---

## 3) Model Artifacts + Metrics

Artifacts in `/Users/josh/Desktop/emo_classifier/artifacts`:
- `emo_smoke.joblib`
- `emo_smoke_offline.joblib`
- `emo_fma_200.joblib`
- `emo_fma_200_eval.json`
- `emo_fma_200_eval_errors.csv`
- `emo_fma_1000_fast.joblib`
- `emo_fma_1000_fast_eval.json`
- `emo_fma_1000_fast_eval_errors.csv`

### Best current checkpoint (quick first pass)
- Model: `artifacts/emo_fma_1000_fast.joblib`
- Eval JSON: `artifacts/emo_fma_1000_fast_eval.json`
- Embedded examples: `999` (skipped: `1`)
- Confusion matrix:
  - `[[87, 13], [20, 80]]`
- Accuracy: `0.835`
- Class behavior:
  - `non_emo` recall stronger than `emo`
  - Some high-confidence false positives remain

### 200-sample run
- Model: `artifacts/emo_fma_200.joblib`
- Accuracy: `0.825`
- Confusion matrix:
  - `[[17, 3], [4, 16]]`

Interpretation:
- Pipeline works end-to-end.
- Current quality is plausible for first-pass labeling, not final production quality.

---

## 4) Uncommitted Work In Progress

Two scripts are complete locally but not committed yet:

1) `scripts/build_relabel_queue.py`
- Input: eval errors CSV
- Output: ranked relabel queue CSV
- Ranks by confidence margin from threshold and adds review action
- Useful for targeted human relabeling pass

2) `scripts/download_jamendo_subset.py`
- Input: label CSV with Jamendo-linked paths
- Parses track IDs and downloads MP3s from:
  - `https://mp3d.jamendo.com/download/track/{track_id}/mp32/`
- Balanced per-class sampling
- Writes downloaded labels CSV + failed CSV

These two files are currently untracked in git.

---

## 5) What Was Already Pushed

Latest pushed commit:
- `c0e7985 Build end-to-end emo classifier pipeline and dataset tooling`

Remote status:
- `main` is on `origin/main` at the same commit
- Everything from the core pipeline is already available in GitHub repo `jram4/emo_classifier`

---

## 6) Resume Plan (Concrete, Execution Order)

When resuming, do this in order:

1) Commit the two pending scripts.
2) Generate a relabel queue from the 1000-fast eval errors.
3) Perform a small manual relabel pass on highest-confidence mistakes.
4) Retrain on corrected labels.
5) Re-evaluate and compare confusion/error profile.
6) Optionally add Jamendo subset audio for domain diversity.

---

## 7) Exact Resume Commands

From `/Users/josh/Desktop/emo_classifier`:

```bash
source .venv/bin/activate
```

### A) Commit/push pending scripts
```bash
git add scripts/build_relabel_queue.py scripts/download_jamendo_subset.py
git commit -m "Add relabel queue builder and Jamendo subset downloader"
git push origin main
```

### B) Build relabel queue from current best eval errors
```bash
python scripts/build_relabel_queue.py \
  --errors-csv artifacts/emo_fma_1000_fast_eval_errors.csv \
  --out-csv data/relabel_queue_fma_1000_top100.csv \
  --top-n 100 \
  --threshold 0.5
```

### C) Retrain after relabel corrections (example)
```bash
emo-classifier train \
  --csv data/labels_fma_train_1000.csv \
  --model-out artifacts/emo_fma_1000_v2.joblib \
  --skip-errors \
  --local-files-only
```

### D) Optional Jamendo subset audio pull (balanced)
```bash
python scripts/download_jamendo_subset.py \
  --labels-csv data/labels_jamendo_candidates.csv \
  --out-audio-root external/jamendo_audio_subset \
  --out-labels-csv data/labels_jamendo_downloaded.csv \
  --max-per-class 250 \
  --seed 42 \
  --sleep-seconds 0.05
```

---

## 8) Known Constraints / Notes
- Spotify Web API supports metadata/discovery, not full-track bulk training audio via standard API paths.
- Existing training path here is built around local audio from FMA/Jamendo-style sources.
- `.gitignore` intentionally excludes:
  - `artifacts/`
  - `external/`
  - most generated `data/*.csv`
  - caches/venv/system temp files

---

## 9) Security Note
- Spotify app credentials were shared in chat context.
- They are intentionally not copied into repository files here.
- Recommended after class demo: rotate client secret.

---

## 10) Quick Health Check
- Repo is in a good resumable state.
- Core pipeline is committed and pushed.
- Dataset + first-pass models exist locally.
- Next highest-impact move is relabeling high-confidence errors and retraining.
