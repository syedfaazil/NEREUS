# HPC Usage Guide — PlanktonAI

The entire pipeline — environment setup, dataset pull, training, and weight upload — runs **inside the PBS compute job**. The only things done on the login node are `scp` and `qsub`.

---

## Prerequisites (done locally, once)

### 1. Fill in `config.py`

```python
# config.py (already gitignored)
HF_TOKEN        = "hf_your_token_here"          # huggingface.co/settings/tokens → Write
HF_DATASET_REPO = "yourname/planktonai-dataset"
HF_MODEL_REPO   = "yourname/planktonai-models"
```

### 2. Prepare and upload the dataset archive

```bash
# On your local machine:
python scripts/prepare_dataset.py --output-archive   # uses RESIZE_TO from config.py
python scripts/upload_prepared.py                    # uses HF_TOKEN + HF_DATASET_REPO
```

---

## Submitting a job

```bash
# 1. Upload code to HPC login node (includes config.py)
scp -r . user@172.17.18.100:~/planktonai/

# 2. Submit — that's it
ssh user@172.17.18.100 "cd ~/planktonai && qsub hpc/submit_job.pbs"
```

---

## What the job does (all inside the compute node)

| Phase | Action | Cost |
|-------|--------|------|
| **1** | Install micromamba to `$HOME/.local/bin` | One-time ~1 min |
| **2** | Create `planktonai` conda env (Python 3.8) | One-time ~3 min |
| **3** | `pip install -r requirements.txt` | One-time ~3 min |
| **4** | Read `config.py` for credentials + hyperparams | Instant |
| **5** | Probe `$TMPDIR` free space → adaptive sample fraction | Instant |
| **6** | `hf_hub_download` prepared_dataset.tar.gz → `$TMPDIR` | ~2-5 min |
| **7** | Extract archive | ~1 min |
| **8** | Train YOLO + MobileNetV3 | Your epoch budget |
| **9** | Upload weights to HF Hub model repo | ~1 min |

> **Phases 1–3 are skipped on every subsequent job** since `$HOME` is persistent. The ~8 min one-time cost is paid only on the very first submission.

---

## Monitor

```bash
qstat -a              # job queue
tail -f training.log  # live log (after job starts)
```

---

## Retrieve weights locally

```bash
python -c "
from huggingface_hub import hf_hub_download
import config as c
hf_hub_download(c.HF_MODEL_REPO, 'yolov5n_best.pt',                    local_dir='models/')
hf_hub_download(c.HF_MODEL_REPO, 'mobilenetv3_classifier_best.pt',     local_dir='models/')
"
```

---

## Override config at submission time

```bash
# Use GPU queue, 100 epochs, 30% pruning
DEVICE=cuda EPOCHS=100 PRUNE_RATIO=0.3 qsub hpc/submit_job.pbs
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `config.py not found` | `scp config.py user@172.17.18.100:~/planktonai/` |
| `HF_TOKEN not set in config.py` | Edit `config.py` — replace `hf_xxx` placeholder |
| `ultralytics not importable` | First job likely failed mid-dep-install; resubmit |
| `TMPDIR full` | Job auto-reduces sample. Check `training.log` for tier. |
| `dos2unix` error | `dos2unix hpc/submit_job.pbs` on login node |
| Want to pre-warm env | Run `bash hpc/setup_login_node.sh` (optional) |

---

## Connection

- **Host**: `172.17.18.100`  
- **Queue**: `workq` (CPU default) — edit `#PBS -q` for GPU queue
