# =============================================================================
# config.example.py — Template for config.py
#
# Copy this file to config.py and fill in your real values.
# config.py is gitignored — this template is tracked instead.
# =============================================================================

HF_TOKEN        = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_DATASET_REPO = "yourname/planktonai-dataset"
HF_RAW_REPO     = "yourname/planktonai-raw"
HF_MODEL_REPO   = "yourname/planktonai-models"

RESIZE_TO        = 224
SAMPLE_FRACTION  = 1.0

EPOCHS           = 50
BATCH_SIZE       = 16
DEVICE           = "cpu"
WORKERS          = 8
PRUNE_RATIO      = 0.0
PRUNE_STRUCTURED = False
PRUNE_EPOCHS     = 5
