from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.request import Request, urlopen

RAW_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{commit}/banking_data"
)
MAX_FILE_BYTES = 25 * 1024 * 1024


def download_bytes(url: str, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    request = Request(url, headers={"User-Agent": "triagedesk-data-fetch/0.1"})
    with urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError(f"refusing download larger than {max_bytes} bytes")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"download exceeded {max_bytes} bytes")
    return content


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_banking77(raw_dir: Path, commit: str = "master") -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": "PolyAI-LDN/task-specific-datasets",
        "commit": commit,
        "files": {},
    }
    for filename in ("train.csv", "test.csv", "categories.json"):
        url = f"{RAW_BASE.format(commit=commit)}/{filename}"
        content = download_bytes(url)
        destination = raw_dir / filename
        with tempfile.NamedTemporaryFile(dir=raw_dir, prefix=f".{filename}.", delete=False) as temp:
            temp.write(content)
            temporary = Path(temp.name)
        temporary.replace(destination)
        manifest["files"][filename] = {
            "url": url,
            "sha256": sha256_bytes(content),
            "bytes": len(content),
        }
    validate_banking77(raw_dir)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"{path} must have exactly text,category columns")
        rows = [{"text": row["text"], "category": row["category"]} for row in reader]
    if not rows or any(not row["text"].strip() or not row["category"].strip() for row in rows):
        raise ValueError(f"{path} contains an empty text or category")
    return rows


def validate_banking77(raw_dir: Path) -> dict[str, int]:
    train = read_rows(raw_dir / "train.csv")
    test = read_rows(raw_dir / "test.csv")
    categories = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    if not isinstance(categories, list) or len(categories) != 77:
        raise ValueError("categories.json must contain the 77 canonical Banking77 categories")
    allowed = set(categories)
    observed = {row["category"] for row in train + test}
    if observed - allowed:
        raise ValueError(f"unknown labels found: {sorted(observed - allowed)}")
    normalized_train = {" ".join(row["text"].casefold().split()) for row in train}
    normalized_test = {" ".join(row["text"].casefold().split()) for row in test}
    leakage = normalized_train & normalized_test
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "categories": len(categories),
        "normalized_train_test_duplicates": len(leakage),
    }


def rows_to_xy(rows: Iterable[dict[str, str]]) -> tuple[list[str], list[str]]:
    rows = list(rows)
    return [row["text"] for row in rows], [row["category"] for row in rows]
