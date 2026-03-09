"""Load and parse ground-truth annotation files for Bessel beam datasets.

Annotation format
-----------------
Each file is a plain-text record of firing events, one entry per line.
Supported line formats (all UTF-8):

1. Frame range + counts:
       [147]:4(paired)+5(single)
       [147-150]:4(paired)+5(single)
       [147]:4+5
       [147]:4

2. Full-width brackets (common in Chinese-locale files):
       （147）:4(paired)+5(single)
       （147-150）:4+5

3. Lines starting with '#' or blank lines are treated as comments and skipped.

Parsed output per entry::

    {
        "frame_start": int,
        "frame_end": int,          # == frame_start for single-frame entries
        "n_paired": int,
        "n_single": int,
        "n_total": int,            # n_paired + n_single
        "raw": str,                # original line (stripped)
    }
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Half-width brackets: [47] or [47-52]
_RE_HW = re.compile(
    r"\[(\d+)(?:-(\d+))?\]"        # [start] or [start-end]
    r"\s*:\s*"
    r"(\d+)"                        # first count (paired or total)
    r"(?:\s*(?:\(.*?\)|paired|单配)?)?"  # optional label in parens
    r"(?:\s*\+\s*(\d+)"            # optional +single
    r"(?:\s*(?:\(.*?\)|single|单)?)?)?"
)

# Full-width brackets: （47）or（47-52）
_RE_FW = re.compile(
    r"[（\uFF08](\d+)(?:[-–](\d+))?[）\uFF09]"
    r"\s*[：:]\s*"
    r"(\d+)"
    r"(?:\s*(?:\(.*?\)|（.*?）|paired|单配)?)?"
    r"(?:\s*[+＋]\s*(\d+)"
    r"(?:\s*(?:\(.*?\)|（.*?）|single|单)?)?)?"
)


def _parse_line(line: str) -> Optional[dict]:
    """Parse one non-comment annotation line.

    Returns a dict or None if the line is unparseable.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    for pattern in (_RE_HW, _RE_FW):
        m = pattern.search(line)
        if m:
            frame_start = int(m.group(1))
            frame_end = int(m.group(2)) if m.group(2) else frame_start
            n_paired = int(m.group(3))
            n_single = int(m.group(4)) if m.group(4) else 0
            return {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "n_paired": n_paired,
                "n_single": n_single,
                "n_total": n_paired + n_single,
                "raw": line,
            }

    logger.debug("Could not parse annotation line: %r", line)
    return None


def load_annotation(txt_path: str | Path) -> list[dict]:
    """Load a ground-truth annotation file.

    Args:
        txt_path: Path to the annotation .txt file (UTF-8 encoded).

    Returns:
        List of parsed annotation dicts, one per valid line.
        Each dict has keys: frame_start, frame_end, n_paired, n_single,
        n_total, raw.

    Raises:
        FileNotFoundError: If txt_path does not exist.
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    entries: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = _parse_line(line)
        if entry is not None:
            entries.append(entry)
        elif line.strip() and not line.strip().startswith("#"):
            logger.debug("Line %d unparseable: %r", lineno, line)

    logger.info("Loaded %d annotation entries from %s", len(entries), path)
    return entries


def annotation_to_frame_set(
    annotation: list[dict],
) -> dict[int, dict]:
    """Convert annotation list to a per-frame index.

    For each frame index covered by any annotation entry, accumulate the
    maximum n_paired and n_single counts.  When a range [t1-t2] spans
    multiple frames, each frame is assumed to have the same counts.

    Args:
        annotation: List of dicts from load_annotation().

    Returns:
        Dict mapping frame_idx -> {"n_paired": int, "n_single": int, "n_total": int}.
    """
    frame_index: dict[int, dict] = {}
    for entry in annotation:
        for f in range(entry["frame_start"], entry["frame_end"] + 1):
            existing = frame_index.get(f)
            if existing is None:
                frame_index[f] = {
                    "n_paired": entry["n_paired"],
                    "n_single": entry["n_single"],
                    "n_total": entry["n_total"],
                }
            else:
                # Take max to handle overlapping entries
                frame_index[f] = {
                    "n_paired": max(existing["n_paired"], entry["n_paired"]),
                    "n_single": max(existing["n_single"], entry["n_single"]),
                    "n_total": max(existing["n_total"], entry["n_total"]),
                }
    return frame_index
