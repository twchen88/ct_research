import json
import hashlib
from typing import Any


def hash_dict(d: dict[str, Any], *, length: int = 8) -> str:
    """
    Compute a deterministic hash of a dictionary.
    - Sorts keys
    - Uses JSON for canonicalization
    """
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]