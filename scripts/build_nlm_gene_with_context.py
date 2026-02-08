#!/usr/bin/env python3
"""
Build context-rich NLM Gene JSONL files from official BioC XML release.

Expected input layout (from NLM-Gene-Corpus.zip):
  Corpus/
    Pmidlist.Train.txt
    Pmidlist.Test.txt
    FINAL/
      <PMID>.BioC.XML

Outputs:
  data/datasets/nlm_gene/train.jsonl.gz
  data/datasets/nlm_gene/test.jsonl.gz

Usage:
  python scripts/build_nlm_gene_with_context.py \
    --corpus-dir data/datasets/nlm_gene/raw/Corpus \
    --output-dir data/datasets/nlm_gene \
    --left-words 32 \
    --right-words 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


INVALID_DB_IDS = {"", "-", "-1", "-000", "-111"}
DB_ID_SPLIT_RE = re.compile(r"[,\|;]")
TOKEN_RE = re.compile(r"\S+")


def _parse_infons(elem: ET.Element) -> Dict[str, str]:
    infons: Dict[str, str] = {}
    for infon in elem.findall("infon"):
        key = infon.attrib.get("key", "").strip()
        value = (infon.text or "").strip()
        if key:
            infons[key] = value
    return infons


def _safe_int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _normalize_gold_ids(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith("-222"):
        raw = raw.lstrip("-222,")
    if raw in INVALID_DB_IDS:
        return []

    ids: List[str] = []
    for token in DB_ID_SPLIT_RE.split(raw):
        db_id = token.strip()
        if not db_id or db_id in INVALID_DB_IDS:
            continue
        if db_id.startswith("NCBIGene:"):
            ids.append(db_id)
        elif db_id.isdigit():
            ids.append(f"NCBIGene:{db_id}")
        else:
            ids.append(db_id)
    # keep order, remove duplicates
    seen = set()
    deduped = []
    for gid in ids:
        if gid in seen:
            continue
        seen.add(gid)
        deduped.append(gid)
    return deduped


def _extract_context(
    passage_text: str,
    mention_text: str,
    start_abs: int,
    end_abs: int,
    passage_offset: int,
    left_words: int,
    right_words: int,
) -> Tuple[str, str]:
    """
    Compute left/right context in passage coordinates.
    Falls back to mention string search when offsets don't align.
    """
    if not passage_text:
        return "", ""

    start_rel = start_abs - passage_offset
    end_rel = end_abs - passage_offset

    valid_bounds = 0 <= start_rel <= end_rel <= len(passage_text)
    if not valid_bounds:
        pos = passage_text.find(mention_text)
        if pos == -1:
            return "", ""
        start_rel = pos
        end_rel = pos + len(mention_text)

    tokens = list(TOKEN_RE.finditer(passage_text))
    if not tokens:
        return "", ""

    mention_start_idx = None
    mention_end_idx = None
    for idx, tok in enumerate(tokens):
        if mention_start_idx is None and tok.end() > start_rel:
            mention_start_idx = idx
        if tok.start() < end_rel:
            mention_end_idx = idx
        if mention_start_idx is not None and tok.start() >= end_rel:
            break

    if mention_start_idx is None:
        mention_start_idx = len(tokens) - 1
    if mention_end_idx is None:
        mention_end_idx = mention_start_idx

    # Left context: up to `left_words` tokens immediately before mention.
    left_start_idx = max(0, mention_start_idx - max(0, left_words))
    left_start_char = tokens[left_start_idx].start() if left_words > 0 and mention_start_idx > 0 else start_rel

    # Right context: up to `right_words` tokens immediately after mention.
    if right_words > 0 and mention_end_idx + 1 < len(tokens):
        right_end_idx = min(len(tokens) - 1, mention_end_idx + right_words)
        right_end_char = tokens[right_end_idx].end()
    else:
        right_end_char = end_rel

    context_left = passage_text[left_start_char:start_rel]
    context_right = passage_text[end_rel:right_end_char]
    return context_left, context_right


def _iter_mentions_from_bioc(
    xml_path: Path,
    split: str,
    left_words: int,
    right_words: int,
) -> Iterable[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for document in root.findall("./document"):
        doc_id = (document.findtext("id") or "").strip()
        if not doc_id:
            continue

        mention_idx = 0
        for passage in document.findall("./passage"):
            passage_text = passage.findtext("text") or ""
            passage_offset = _safe_int(passage.findtext("offset"), default=0)

            for ann in passage.findall("./annotation"):
                infons = _parse_infons(ann)
                gold_ids = _normalize_gold_ids(infons.get("NCBI Gene identifier", ""))
                if not gold_ids:
                    continue

                ann_text = (ann.findtext("text") or "").strip()
                if not ann_text:
                    continue

                loc = ann.find("./location")
                if loc is None:
                    continue

                start_abs = _safe_int(loc.attrib.get("offset"), default=-1)
                length = _safe_int(loc.attrib.get("length"), default=0)
                if start_abs < 0 or length <= 0:
                    continue
                end_abs = start_abs + length

                context_left, context_right = _extract_context(
                    passage_text=passage_text,
                    mention_text=ann_text,
                    start_abs=start_abs,
                    end_abs=end_abs,
                    passage_offset=passage_offset,
                    left_words=left_words,
                    right_words=right_words,
                )

                mention_idx += 1
                yield {
                    "dataset": "nlm_gene",
                    "split": split,
                    "ontology_key": "entrez",
                    "doc_id": doc_id,
                    "mention_id": f"{doc_id}.{mention_idx}",
                    "mention": ann_text,
                    "context_left": context_left,
                    "context_right": context_right,
                    "start": start_abs,
                    "end": end_abs,
                    "gold_ids": gold_ids,
                }


def _load_pmid_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_split(
    corpus_dir: Path,
    split: str,
    output_path: Path,
    left_words: int,
    right_words: int,
) -> None:
    pmid_file = corpus_dir / ("Pmidlist.Train.txt" if split == "train" else "Pmidlist.Test.txt")
    final_dir = corpus_dir / "FINAL"
    pmids = _load_pmid_list(pmid_file)

    total_docs = 0
    total_mentions = 0
    context_nonempty = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as out_f:
        for pmid in pmids:
            xml_path = final_dir / f"{pmid}.BioC.XML"
            if not xml_path.exists():
                continue
            total_docs += 1

            for record in _iter_mentions_from_bioc(
                xml_path,
                split=split,
                left_words=left_words,
                right_words=right_words,
            ):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_mentions += 1
                if (record["context_left"] or "").strip() or (record["context_right"] or "").strip():
                    context_nonempty += 1

    pct = (100.0 * context_nonempty / total_mentions) if total_mentions else 0.0
    print(
        f"[{split}] docs={total_docs} mentions={total_mentions} "
        f"context_nonempty={context_nonempty} ({pct:.2f}%) -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context-rich NLM Gene JSONL from BioC XML")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/datasets/nlm_gene/raw/Corpus"),
        help="Path to extracted Corpus directory from NLM-Gene-Corpus.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/nlm_gene"),
        help="Directory where output .jsonl.gz files will be written",
    )
    parser.add_argument(
        "--left-words",
        type=int,
        default=32,
        help="Number of words to include before mention (default: 32)",
    )
    parser.add_argument(
        "--right-words",
        type=int,
        default=32,
        help="Number of words to include after mention (default: 32)",
    )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    _build_split(
        corpus_dir=corpus_dir,
        split="train",
        output_path=args.output_dir / "train.jsonl.gz",
        left_words=args.left_words,
        right_words=args.right_words,
    )
    _build_split(
        corpus_dir=corpus_dir,
        split="test",
        output_path=args.output_dir / "test.jsonl.gz",
        left_words=args.left_words,
        right_words=args.right_words,
    )


if __name__ == "__main__":
    main()
