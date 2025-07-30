# src/logging_config.py
import logging
import sys
import re
from pathlib import Path

# Import the config variables needed for the filename and settings
from src import config

# Define a new directory for run-specific logs
LOGS_DIR = config.PROJECT_ROOT / "logs"

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename."""
    # Replace slashes/colons in model names with a hyphen
    name = name.replace("/", "-").replace(":", "_")
    # Remove any other characters that are invalid in filenames
    name = re.sub(r'[\\*?:"<>|]', "", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Truncate to a reasonable length to avoid OS limits
    return name[:100]

def setup_run_logging(query: str):
    """
    Sets up logging for a specific pipeline run, creating a unique log file.

    - Creates a log file in the `logs/` directory.
    - The filename is composed of the query and the model from config.
    - Continues to print logs to the console as before.
    """
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)

    # Clear any existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- 1. Create the Dynamic Log Filename ---
    sanitized_query = sanitize_filename(query)
    
    # Get the appropriate model name based on the PIPELINE config
    if config.PIPELINE == "ollama":
        model_name = sanitize_filename(config.OLLAMA_SELECTOR_MODEL_NAME)
    elif config.PIPELINE == "gemini":
        model_name = sanitize_filename(config.GEMINI_SELECTOR_MODEL_NAME)
    elif config.PIPELINE == "huggingface":
        model_name = sanitize_filename(config.HF_SELECTOR_MODEL_ID)
    else:
        model_name = "unknown_model"

    log_filename = f"run_{sanitized_query}_{model_name}.log"
    
    # Ensure the logs/ directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    log_filepath = LOGS_DIR / log_filename

    # --- 2. Configure Handlers (File and Console) ---
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )

    # Create a file handler that writes to our new dynamic file
    # Use 'w' mode to create a fresh log for each run
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler to also print logs to the terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(config.LOG_LEVEL) # You could set this to logging.INFO for a cleaner console
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- 3. Quiet Down Noisy Libraries ---
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("whoosh.index").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # --- 4. Log the Initial Status ---
    logging.info(f"Logging configured to level {config.LOG_LEVEL}. Console output enabled.")
    logging.info(f"Saving run-specific log to: {log_filepath}")