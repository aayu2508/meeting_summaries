# utils/common.py
import hashlib
import re
from pathlib import Path
from typing import Any, Dict

# Root for all meeting outputs, used by helpers below
DATA_ROOT = Path("data/outputs")

def norm_key(s: str) -> str:
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def canonical_idea_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[-_]", " ", s)
    return s

def load_kv_file(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        meta[k] = v
    return meta

def get_meeting_base_dir(meeting_id: str) -> Path:
    return DATA_ROOT / meeting_id

def load_metadata(base_dir: Path) -> Dict[str, Any]:
    meta_txt = base_dir / "metadata.txt"
    if meta_txt.exists():
        return load_kv_file(meta_txt)
    return {}
