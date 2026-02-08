#!/usr/bin/env python3
"""
Build context-rich NCBI Disease JSONL files from the official corpus text files.

Expected input layout (from the NCBI Disease Corpus release):
  <corpus-dir>/
    NCBItrainset_corpus.txt
    NCBIdevelopset_corpus.txt
    NCBItestset_corpus.txt

Outputs:
  data/datasets/ncbi_disease/train.jsonl.gz
  data/datasets/ncbi_disease/validation.jsonl.gz
  data/datasets/ncbi_disease/test.jsonl.gz

Usage:
  python scripts/build_ncbi_disease_with_context.py \
    --corpus-dir data/datasets/ncbi_disease/raw \
    --output-dir data/datasets/ncbi_disease \
    --left-words 32 \
    --right-words 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TOKEN_RE = re.compile(r"\S+")
ID_SPLIT_RE = re.compile(r"[|+,;]")
INVALID_DB_IDS = {"", "-", "-1", "NIL", "N/A"}


def _safe_int(value: str, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_gold_ids(raw: str) -> List[str]:
    ids: List[str] = []
    for token in ID_SPLIT_RE.split((raw or "").strip()):
        db_id = token.strip()
        if not db_id or db_id in INVALID_DB_IDS:
            continue

        if db_id.startswith(("MESH:", "OMIM:")):
            norm = db_id
        elif len(db_id) > 1 and db_id[0] in {"C", "D"} and db_id[1:].isdigit():
            norm = f"MESH:{db_id}"
        else:
            norm = db_id
        ids.append(norm)

    # Keep order, remove duplicates.
    seen = set()
    deduped = []
    for gid in ids:
        if gid in seen:
            continue
        seen.add(gid)
        deduped.append(gid)
    return deduped


def _iter_blocks(path: Path) -> Iterable[List[str]]:
    block: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if block:
                    yield block
                    block = []
                continue
            block.append(line)
    if block:
        yield block


def _extract_context(
    text: str,
    start: int,
    end: int,
    left_words: int,
    right_words: int,
) -> Tuple[str, str]:
    if not text:
        return "", ""

    n = len(text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))

    tokens = list(TOKEN_RE.finditer(text))
    if not tokens:
        return "", ""

    mention_token_idxs = [
        i for i, tok in enumerate(tokens)
        if not (tok.end() <= start or tok.start() >= end)
    ]

    if mention_token_idxs:
        mention_start_idx = mention_token_idxs[0]
        mention_end_idx = mention_token_idxs[-1]
    else:
        # Fallback when offsets do not align perfectly.
        mention_start_idx = len(tokens) - 1
        for i, tok in enumerate(tokens):
            if tok.end() > start:
                mention_start_idx = i
                break
        mention_end_idx = mention_start_idx

    if left_words > 0 and mention_start_idx > 0:
        left_start_idx = max(0, mention_start_idx - left_words)
        left_start_char = tokens[left_start_idx].start()
    else:
        left_start_char = start

    if right_words > 0 and mention_end_idx + 1 < len(tokens):
        right_end_idx = min(len(tokens) - 1, mention_end_idx + right_words)
        right_end_char = tokens[right_end_idx].end()
    else:
        right_end_char = end

    context_left = text[left_start_char:start]
    context_right = text[end:right_end_char]
    return context_left, context_right


def _parse_block(block: List[str]) -> Optional[Dict]:
    doc_id = ""
    title = ""
    abstract = ""
    annotations: List[Tuple[int, int, str, str]] = []

    for line in block:
        if "|t|" in line:
            parts = line.split("|t|", 1)
            if len(parts) != 2:
                continue
            doc_id = parts[0].strip()
            title = parts[1]
            continue
        if "|a|" in line:
            parts = line.split("|a|", 1)
            if len(parts) != 2:
                continue
            if not doc_id:
                doc_id = parts[0].strip()
            abstract = parts[1]
            continue

        cols = line.split("\t")
        if len(cols) < 6:
            continue
        if not doc_id:
            doc_id = cols[0].strip()
        start = _safe_int(cols[1], default=-1)
        end = _safe_int(cols[2], default=-1)
        mention = cols[3].strip()
        db_ids = cols[5].strip()
        if start < 0 or end <= start or not mention:
            continue
        annotations.append((start, end, mention, db_ids))

    if not doc_id:
        return None

    full_text = f"{title} {abstract}".strip() if title and abstract else (title or abstract)
    return {
        "doc_id": doc_id,
        "text": full_text,
        "annotations": annotations,
    }


def _build_split(
    corpus_file: Path,
    split: str,
    output_path: Path,
    left_words: int,
    right_words: int,
) -> None:
    if not corpus_file.exists():
        raise FileNotFoundError(f"Missing corpus file for split '{split}': {corpus_file}")

    doc_count = 0
    mention_count = 0
    context_nonempty = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as out_f:
        for block in _iter_blocks(corpus_file):
            doc = _parse_block(block)
            if not doc:
                continue
            doc_count += 1
            mention_idx = 0

            for start, end, mention, db_ids in doc["annotations"]:
                gold_ids = _normalize_gold_ids(db_ids)
                if not gold_ids:
                    continue

                context_left, context_right = _extract_context(
                    text=doc["text"],
                    start=start,
                    end=end,
                    left_words=left_words,
                    right_words=right_words,
                )
                mention_idx += 1

                record = {
                    "dataset": "ncbi_disease",
                    "split": split,
                    "ontology_key": "ctd_diseases",
                    "doc_id": doc["doc_id"],
                    "mention_id": f"{doc['doc_id']}.{mention_idx}",
                    "mention": mention,
                    "context_left": context_left,
                    "context_right": context_right,
                    "start": start,
                    "end": end,
                    "gold_ids": gold_ids,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                mention_count += 1
                if (context_left or "").strip() or (context_right or "").strip():
                    context_nonempty += 1

    pct = (100.0 * context_nonempty / mention_count) if mention_count else 0.0
    print(
        f"[{split}] docs={doc_count} mentions={mention_count} "
        f"context_nonempty={context_nonempty} ({pct:.2f}%) -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build context-rich NCBI Disease JSONL files from corpus text files"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/datasets/ncbi_disease/raw"),
        help="Directory containing NCBI*set_corpus.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/ncbi_disease"),
        help="Directory where output .jsonl.gz files will be written",
    )
    parser.add_argument(
        "--left-words",
        type=int,
        default=32,
        help="Number of words before mention (default: 32)",
    )
    parser.add_argument(
        "--right-words",
        type=int,
        default=32,
        help="Number of words after mention (default: 32)",
    )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    split_files = {
        "train": corpus_dir / "NCBItrainset_corpus.txt",
        "validation": corpus_dir / "NCBIdevelopset_corpus.txt",
        "test": corpus_dir / "NCBItestset_corpus.txt",
    }

    for split, corpus_file in split_files.items():
        output_path = args.output_dir / f"{split}.jsonl.gz"
        _build_split(
            corpus_file=corpus_file,
            split=split,
            output_path=output_path,
            left_words=args.left_words,
            right_words=args.right_words,
        )


if __name__ == "__main__":
    main()
