# src/evaluation/evaluate_craft_chebi.py
import argparse
import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.asyncio import tqdm_asyncio

from src import config
from src.pipeline import RAGPipeline
from src.pipeline import create_pipeline
from src.utils.cache import load_cache, save_cache
from src.utils.logging_config import setup_run_logging
from src.utils.ontology_utils import uri_to_curie
from src.utils.token_tracker import token_tracker
from src.utils.context_window import make_context_window

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger: logging.Logger = logging.getLogger(__name__)

# Defaults
DEFAULT_BIONLP_DIR: Path = PROJECT_ROOT / "data" / "CRAFT_bionlp" / "chebi"
DEFAULT_EVAL_OUT: Path = PROJECT_ROOT / f"evaluation_results_CRAFT_CHEBI_{config.PIPELINE}.json"
DEFAULT_SYSTEM_OUTPUT_BASE: Path = PROJECT_ROOT / "data" / "CRAFT_system_output"
DEFAULT_WRITE_SYSTEM_OUTPUT: bool = True
DEFAULT_DEDUPE_SURFACE_FORMS: bool = True

_BIONLP_LINE = re.compile(r"^(T\d+)\t([^\s]+)\s+([0-9; ]+)\t(.*)$")


def normalize_to_curie(identifier: Optional[str]) -> Optional[str]:
    """
    Normalize a predicted identifier into a CHEBI CURIE (e.g., 'CHEBI:1234').
    Accepts CHEBI URIs, 'CHEBI_1234', or already normalized 'CHEBI:1234'.
    """
    if not identifier:
        return None
    ident = identifier.strip()
    if ident.startswith(("http://", "https://")):
        curie = uri_to_curie(ident, config.CURIE_PREFIX_MAP)
        if curie:
            return curie
    if ident.upper().startswith("CHEBI_"):
        return "CHEBI:" + ident.split("_", 1)[1]
    return ident


def _unpack_result(obj: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Unpack heterogeneous pipeline results into (final_result, candidates).

    Supports:
      - (final_result, candidates)
      - [final_result, candidates]
      - {'final_result': {...}, 'candidates': [...]}
      - final_result dict alone (with 'id'/'curie'/'identifier')
      - {'result': {...}, 'candidates': [...]}
    """
    if not obj:
        return None, []

    if isinstance(obj, (list, tuple)):
        if not obj:
            return None, []
        first = obj[0]
        cands = obj[1] if len(obj) > 1 and isinstance(obj[1], (list, tuple)) else []
        return (first if isinstance(first, dict) else None), list(cands)

    if isinstance(obj, dict):
        if "final_result" in obj:
            return obj.get("final_result"), obj.get("candidates") or []
        if any(k in obj for k in ("id", "curie", "identifier", "label")):
            # Sometimes the final result itself was cached directly.
            cands = obj.get("candidates")
            return obj, (cands if isinstance(cands, list) else [])
        if "result" in obj and isinstance(obj["result"], dict):
            return obj["result"], obj.get("candidates") or []

    return None, []


def parse_craft_bionlp_dir(
    bionlp_dir: Path,
    doc_whitelist: Optional[Set[str]] = None,
    dedupe_surface_forms: bool = DEFAULT_DEDUPE_SURFACE_FORMS,
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Parse CRAFT CHEBI BioNLP files into a list of gold items.

    Each item:
      {
        'doc_id': str,
        'ann_id': str,
        'offsets': str,        # "start end" or "start end;start end"
        'text': str,           # surface form
        'true_curies': [str],  # ['CHEBI:xxxx']
        'document_text': str,  # NEW: full document text for context
        'start_offset': int,   # NEW: parsed start offset
        'end_offset': int,     # NEW: parsed end offset
      }

    Parameters
    ----------
    bionlp_dir : Path
        Directory with *.bionlp gold files where TYPE is 'CHEBI:xxxx'.
    doc_whitelist : Optional[Set[str]]
        If provided, include only these document IDs (stems).
    dedupe_surface_forms : bool
        If True, evaluate unique surface forms across the entire selection.
    limit : Optional[int]
        If provided, randomly cap the number of items after filtering/dedupe.
    seed : int
        RNG seed for reproducible sampling when limit is set.

    Returns
    -------
    List[Dict[str, Any]]
        Parsed gold items.
    """
    print(f"Parsing CRAFT BioNLP from: {bionlp_dir}")
    if not bionlp_dir.exists():
        logger.error("CRAFT BioNLP directory not found: %s", bionlp_dir)
        return []

    # Look for corresponding text files directory
    text_dir = bionlp_dir.parent / "txt"
    if not text_dir.exists():
        # Try alternative location
        text_dir = bionlp_dir.parent.parent / "txt"
    
    if not text_dir.exists():
        logger.warning(f"No text files directory found. Context will be empty. Looked in: {text_dir}")

    items: List[Dict[str, Any]] = []
    seen_text: Set[str] = set()

    files = sorted(bionlp_dir.glob("*.bionlp"))
    if doc_whitelist:
        files = [p for p in files if p.stem in doc_whitelist]
        print(f"Whitelisted docs: {len(files)} files")

    for bionlp_file in files:
        doc_id = bionlp_file.stem
        
        # Load corresponding document text
        doc_text = ""
        txt_file = text_dir / f"{doc_id}.txt"
        if txt_file.exists():
            try:
                with txt_file.open("r", encoding="utf-8") as fh:
                    doc_text = fh.read()
            except Exception as e:
                logger.warning(f"Failed to load text file {txt_file}: {e}")
        else:
            logger.warning(f"Text file not found: {txt_file}")

        with bionlp_file.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if not line or not line.startswith("T"):
                    continue
                m = _BIONLP_LINE.match(line)
                if not m:
                    continue
                ann_id, ann_type, offsets, surface = m.groups()
                if not ann_type.upper().startswith("CHEBI:"):
                    continue

                surface_norm = surface.strip()
                if dedupe_surface_forms:
                    key = surface_norm.lower()
                    if key in seen_text:
                        continue
                    seen_text.add(key)

                # Parse offsets - handle both single and discontinuous spans
                # Format: "start end" or "start end;start2 end2"
                offset_parts = offsets.split(";")
                try:
                    # Use the first span for context generation
                    first_span = offset_parts[0].strip().split()
                    start_offset = int(first_span[0])
                    end_offset = int(first_span[1])
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse offsets '{offsets}' for {doc_id}:{ann_id}: {e}")
                    start_offset = None
                    end_offset = None

                items.append(
                    {
                        "doc_id": doc_id,
                        "ann_id": ann_id,
                        "offsets": offsets,
                        "text": surface_norm,
                        "true_curies": [ann_type],
                        "document_text": doc_text,
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                    }
                )

    if limit is not None and len(items) > limit:
        rnd = random.Random(seed)
        rnd.shuffle(items)
        items = items[:limit]
        print(f"Applied random cap: limit={limit}, seed={seed}")

    print(f"Parsed {len(items)} items")
    return items


async def evaluate_full_pipeline(
    pipeline: RAGPipeline,
    gold_data: List[Dict[str, Any]],
    cache: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Tuple[float, int, int, int, int, List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Evaluate the pipeline on gold surface forms with gold spans (linking-only).

    Parameters
    ----------
    pipeline : BaseRAGPipeline
        Initialized pipeline implementing .run(query=..., semaphore=...).
    gold_data : List[Dict[str, Any]]
        Items as returned by parse_craft_bionlp_dir.
    cache : Dict[str, Any]
        Persistent cache mapping query string -> pipeline result object.
    semaphore : asyncio.Semaphore
        Concurrency control for async runs.

    Returns
    -------
    Tuple containing:
      accuracy : float
      total : int
      hits : int
      cache_hits : int
      failures : int
      misses : List[Dict[str, Any]]
      per_doc_predictions : Dict[str, List[Dict[str, Any]]]
    """
    total = len(gold_data)
    if total == 0:
        return 0.0, 0, 0, 0, 0, [], {}

    hits = 0
    cache_hits = 0
    failures = 0
    misses: List[Dict[str, Any]] = []
    to_run: List[asyncio.Future] = []
    queries: List[str] = []
    contexts: List[str] = []

    for item in gold_data:
        q = item["text"]
        
        # Generate context window for this mention
        mention = item["text"]
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        doc_text = item.get("document_text", "")
        
        context = ""
        if doc_text and start_offset is not None and end_offset is not None:
            try:
                context = make_context_window(doc_text, mention, start_offset, end_offset, radius=100)
            except Exception as e:
                logger.warning(f"Failed to generate context for '{mention}' in {item['doc_id']}: {e}")
                context = ""
        
        # Create cache key that includes context for more precise caching
        cache_key = f"{q}|||{context[:200]}" if context else q  # Truncate context for cache key
        
        if cache_key in cache:
            cache_hits += 1
        else:
            to_run.append(
                pipeline.run(
                    query=q,
                    context=context,  # NEW: Pass context to pipeline
                    semaphore=semaphore,
                    target_ontologies=["chebi"],
                )
            )
            queries.append(cache_key)
            contexts.append(context)

    if to_run:
        results = await tqdm_asyncio.gather(*to_run, desc="Evaluating Pipeline")
        for cache_key, res in zip(queries, results):
            fr, _ = _unpack_result(res)
            if fr:
                cache[cache_key] = res

    per_doc_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for item in gold_data:
        q = item["text"]
        true_ids = set(item["true_curies"])
        
        # Reconstruct cache key
        mention = item["text"]
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        doc_text = item.get("document_text", "")
        
        context = ""
        if doc_text and start_offset is not None and end_offset is not None:
            try:
                context = make_context_window(doc_text, mention, start_offset, end_offset, radius=100)
            except Exception:
                context = ""
        
        cache_key = f"{q}|||{context[:200]}" if context else q
        res = cache.get(cache_key)
        final_result, cand_list = _unpack_result(res)

        pred_curie: Optional[str] = None
        if not final_result:
            failures += 1
        else:
            pred_curie = normalize_to_curie(
                final_result.get("id")
                or final_result.get("curie")
                or final_result.get("identifier")
            )

        ok = bool(pred_curie in true_ids) if pred_curie else False
        if ok:
            hits += 1
        else:
            chosen = pred_curie if pred_curie else None
            conf = float(final_result.get("confidence_score", -1.0)) if final_result else "N/A"

            # Extract both selector and scorer explanations
            selector_explanation = str(final_result.get("selector_explanation", "N/A")) if final_result else "N/A"
            scorer_explanation = str(final_result.get("scorer_explanation", "N/A")) if final_result else "N/A"
            suggested_alternatives = final_result.get("suggested_alternatives", []) if final_result else []
            
            cand_ids: List[str] = []
            if cand_list:
                for c in cand_list:
                    if isinstance(c, dict) and "id" in c:
                        norm = normalize_to_curie(str(c.get("id")))
                        if norm:
                            cand_ids.append(norm)

            misses.append(
                {
                    "query": q,
                    "context": context[:500] if context else "",  # NEW: Include context in miss analysis
                    "chosen_curie": chosen,
                    "true_curies": list(true_ids),
                    "selector_explanation": selector_explanation,
                    "scorer_explanation": scorer_explanation,
                    "suggested_alternatives": suggested_alternatives,
                    "confidence_score": conf,
                    "candidates_provided": cand_ids,
                }
            )

        per_doc_predictions.setdefault(item["doc_id"], []).append(
            {
                "offsets": item["offsets"],
                "text": item["text"],
                "pred_curie": pred_curie,
                "ok": ok,
            }
        )

    valid_attempts = total - failures
    accuracy = (hits / valid_attempts) if valid_attempts > 0 else 0.0
    return accuracy, total, hits, cache_hits, failures, misses, per_doc_predictions


def write_bionlp_system_output(
    per_doc_predictions: Dict[str, List[Dict[str, Any]]],
    out_base: Path,
) -> None:
    """
    Write system predictions in BioNLP format for the official CRAFT scorer.
    Only predictions with a non-empty concept ID are written.
    """
    chebi_out = out_base / "CHEBI"
    chebi_out.mkdir(parents=True, exist_ok=True)

    for doc_id, preds in per_doc_predictions.items():
        lines: List[str] = []
        t_idx = 1
        for p in preds:
            pred = p.get("pred_curie")
            if not pred:
                continue
            lines.append(f"T{t_idx}\t{pred} {p['offsets']}\t{p['text']}")
            t_idx += 1

        out_file = chebi_out / f"{doc_id}.bionlp"
        with out_file.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    ap = argparse.ArgumentParser(description="Evaluate NEL on CRAFT (CHEBI)")
    ap.add_argument(
        "--bionlp-dir",
        type=Path,
        default=DEFAULT_BIONLP_DIR,
        help="Path to CRAFT CHEBI BioNLP files (default: data/CRAFT_bionlp/chebi)",
    )
    ap.add_argument(
        "--docs",
        type=str,
        default="",
        help="Comma-separated PMC doc IDs to include (e.g., PMC2768288,PMC2783567)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of (deduped) items to evaluate",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --limit sampling")
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Evaluate per-mention (no surface-form dedupe)",
    )
    ap.add_argument(
        "--no-system-output",
        action="store_true",
        help="Skip writing BioNLP system output",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override config.MAX_CONCURRENT_REQUESTS for quick smoke tests",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Override cache path (e.g., data/pipeline_cache_craft_dev.json)",
    )
    return ap


async def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    setup_run_logging("evaluation_craft_chebi")

    write_system_output: bool = not args.no_system_output
    dedupe_surface_forms: bool = not args.no_dedupe
    max_conc: int = args.max_concurrency if args.max_concurrency else int(config.MAX_CONCURRENT_REQUESTS)

    doc_whitelist: Optional[Set[str]] = None
    if args.docs.strip():
        doc_whitelist = {d.strip() for d in args.docs.split(",") if d.strip()}

    print(f"Evaluating pipeline '{config.PIPELINE}' on CRAFT/CHEBI subset")
    print(f"  Directory: {args.bionlp_dir}")
    print(f"  Documents: {sorted(doc_whitelist) if doc_whitelist else 'ALL'}")
    print(f"  Limit: {args.limit}")
    print(f"  Seed: {args.seed}")
    print(f"  Dedupe surface forms: {dedupe_surface_forms}")
    print(f"  Write system output: {write_system_output}")
    print(f"  Max concurrency: {max_conc}")
    print(f"  Cache: {args.cache if args.cache else 'DISABLED'}")

    gold = parse_craft_bionlp_dir(
        args.bionlp_dir,
        doc_whitelist=doc_whitelist,
        dedupe_surface_forms=dedupe_surface_forms,
        limit=args.limit,
        seed=args.seed,
    )
    if not gold:
        logger.error("No gold items parsed. Exiting.")
        return

    # ✅ Use cache only if explicitly given
    if args.cache:
        cache_path: Path = args.cache
        cache: Dict[str, Any] = load_cache(cache_path)
    else:
        cache_path = None
        cache: Dict[str, Any] = {}

    try:
        pipeline: RAGPipeline = create_pipeline(config.PIPELINE)
        semaphore = asyncio.Semaphore(max_conc)

        (
            accuracy,
            total,
            correct,
            cache_hits,
            failures,
            misses,
            per_doc,
        ) = await evaluate_full_pipeline(pipeline, gold, cache, semaphore)

        print("\n--- Evaluation Complete ---")
        print(f"Total gold items: {total}")
        print(f"Cache hits: {cache_hits}")
        print(f"Pipeline failures (no result): {failures}")
        valid = total - failures
        print(f"Valid attempts: {valid}")
        print(f"Correct (hits): {correct}")
        if valid > 0:
            print(f"Micro-accuracy: {accuracy:.4f}")
        else:
            print("Micro-accuracy: N/A (no valid attempts)")

        with DEFAULT_EVAL_OUT.open("w", encoding="utf-8") as fh:
            json.dump(misses, fh, indent=2)
        print(f"Wrote {len(misses)} misses → {DEFAULT_EVAL_OUT}")

        if write_system_output:
            write_bionlp_system_output(per_doc, DEFAULT_SYSTEM_OUTPUT_BASE)
            print(f"Wrote system output → {DEFAULT_SYSTEM_OUTPUT_BASE / 'CHEBI'}")

    except Exception as e:
        logger.error("Evaluation error: %s", e, exc_info=True)
    finally:
        print(token_tracker.report_usage())
        try:
            pipeline.close()
        except Exception:
            pass
        # ✅ Only save if a cache path was explicitly given
        if cache_path:
            save_cache(cache_path, cache)

if __name__ == "__main__":
    asyncio.run(main())