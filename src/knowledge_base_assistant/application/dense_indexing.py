import hashlib
import json
from pathlib import Path


_BLOCK_SIZE = 64 * 1024


def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(_BLOCK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()