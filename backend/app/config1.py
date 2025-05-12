
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from YAML
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

try:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logger.error(f"Configuration file not found: {CONFIG_PATH}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML file: {e}")
    raise

# Configuration
MODEL_PATH = Path(config.get("model_path", "models/audio_model")).resolve()
TEST_DATA_DIR = Path(config.get("test_data_dir", "data/test_data")).resolve()

# Validate MODEL_PATH
if not MODEL_PATH.exists() or not MODEL_PATH.is_dir():
    logger.error(f"MODEL_PATH does not exist or is not a directory: {MODEL_PATH}")
    raise ValueError(f"Invalid MODEL_PATH: {MODEL_PATH}")

# Ensure test_data directory exists
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Convert paths to strings for compatibility
MODEL_PATH = str(MODEL_PATH)
TEST_DATA_DIR = str(TEST_DATA_DIR)

# Log resolved paths
logger.info(f"Resolved MODEL_PATH: {MODEL_PATH}")
logger.info(f"Resolved TEST_DATA_DIR: {TEST_DATA_DIR}")