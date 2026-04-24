from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "compressed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compression defaults
DEFAULT_CLUSTERS = 32
DEFAULT_MAX_ITER = 300
DEFAULT_RANDOM_STATE = 42

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]

# JPEG optimization
JPEG_QUALITY = 85

# Batch mode
BATCH_CLUSTERS = [256,128,64,32,16]
