import logging
logger = logging.getLogger("address_extractor")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    "extraction.log"
)
formatter = logging.Formatter(
    "%(asctime)s - %(message)s"
)
file_handler.setFormatter(
    formatter
)
logger.addHandler(
    file_handler
)
