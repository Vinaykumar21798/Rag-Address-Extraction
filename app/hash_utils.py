import hashlib
import re

def generate_sha256(content):
    return hashlib.sha256(content).hexdigest()

def generate_content_hash(text):
    normalized = text.lower()
    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()
    return hashlib.sha256(
        normalized.encode()
    ).hexdigest()

