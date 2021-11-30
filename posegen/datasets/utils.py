import hashlib
from pathlib import Path


def get_md5(path: Path) -> str:
    return hashlib.md5(open(path, 'rb').read()).hexdigest()
