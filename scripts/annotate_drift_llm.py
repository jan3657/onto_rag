#!/usr/bin/env python
import os
import json
import argparse
import logging
import asyncio
from typing import Dict, Any, List

from src import config
from src.infrastructure.llm.clients.gemini import GeminiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift_annotator_gemini")

SYSTEM_AND_TASK_PROMPT = """You are an ontology adjudicator.

Given:
- a query string,
- a single chosen ontology term (FoodOntoRAG’s prediction),
- a list of dataset ground_truth_terms,

you must:
1. Select exactly one "gold_best" term from ground_truth_terms: the term that best matches the intended meaning and specificity of the query.
2. Classify the relationship between the chosen term and this gold_best term using the schema below.
3. Return a single JSON object strictly following the required schema.
4. Be concise and deterministic. Do NOT include chain-of-thought. Do NOT invent external facts.

### STEP 1: Choose gold_best
From "ground_truth_terms", pick the single best gold term for the query:
- Prefer the term whose label/definition/synonyms most directly match the query string.
- Prefer more specific matches over overly generic ones when the query is specific.
- If several are equivalent, choose the one most appropriate for typical recipe/ingredient usage.
- If still tied, choose the one closest in meaning to the chosen_term.

### STEP 2: Assign a drift class (mutually exclusive; choose ONE)
Compare chosen_term vs gold_best and assign exactly one:

- EXACT_MATCH
- CLASS_VS_TAXON
- HIERARCHY_DRIFT
- SYNONYM_OR_LEXICAL
- CROSS_ONTOLOGY_EQUIVALENT
- DATASET_ANNOTATION_ERROR
- MODEL_INCORRECT
- OTHER

### STEP 3: Direction
- CLASS_VS_TAXON: "Taxon→Product" or "Product→Taxon"
- HIERARCHY_DRIFT: "Parent→Child", "Child→Parent", or "Co-hyponyms"
- Otherwise: null

### ROLE HEURISTICS
- NCBITaxon:*  → role = "Taxon"
- FOODON:*     → role = "Product"
- HANSARD:*    → role = "Product"
- UBERON:*     → role = "Product"
(Override with labels/definitions if clearly contradictory.)

### PRIORITIZATION
1. Pick gold_best.
2. If gold_best is clearly wrong for the query → DATASET_ANNOTATION_ERROR.
3. Else if chosen_term clearly mismatches the query → MODEL_INCORRECT.
4. Else apply the most specific fitting class:
   EXACT_MATCH >
   CLASS_VS_TAXON / HIERARCHY_DRIFT / CROSS_ONTOLOGY_EQUIVALENT >
   SYNONYM_OR_LEXICAL >
   OTHER.

### OUTPUT SCHEMA (JSON ONLY)
Return ONLY:

{
  "class": "EXACT_MATCH | CLASS_VS_TAXON | HIERARCHY_DRIFT | SYNONYM_OR_LEXICAL | CROSS_ONTOLOGY_EQUIVALENT | DATASET_ANNOTATION_ERROR | MODEL_INCORRECT | OTHER",
  "direction": "Taxon→Product | Product→Taxon | Parent→Child | Child→Parent | Co-hyponyms | null",
  "chosen": {
    "curie": "<chosen_term.curie>",
    "label": "<chosen_term.label>"
  },
  "gold_best": {
    "curie": "<selected best gold curie or null if none is valid>",
    "label": "<selected best gold label or null if none is valid>"
  },
  "reason": "<ONE short sentence based only on the given data>",
  "confidence": "High | Medium | Low"
}

Now adjudicate the following case and output ONLY the JSON object, nothing else.

CASE:
"""


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first top-level JSON object from the model output.
    """
    s = (text or "").strip()
    # Try direct parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: scan for first balanced { ... }
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break
        start = s.find("{", start + 1)

    raise ValueError(f"Could not extract JSON from response: {text!r}")


def build_minimal_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only the adjudication-relevant fields:
    - query
    - chosen_term
    - ground_truth_terms
    """
    minimal: Dict[str, Any] = {
        "query": case.get("query"),
        "chosen_term": None,
        "ground_truth_terms": []
    }

    chosen = case.get("chosen_term")
    if isinstance(chosen, dict):
        minimal["chosen_term"] = {
            "curie": chosen.get("curie"),
            "label": chosen.get("label"),
            "definition": chosen.get("definition"),
            "synonyms": chosen.get("synonyms", []),
        }

    gts = case.get("ground_truth_terms") or []
    if isinstance(gts, list):
        for gt in gts:
            if not isinstance(gt, dict):
                continue
            minimal["ground_truth_terms"].append(
                {
                    "curie": gt.get("curie"),
                    "label": gt.get("label"),
                    "definition": gt.get("definition"),
                    "synonyms": gt.get("synonyms", []),
                }
            )

    return minimal


async def call_gemini(client: GeminiClient, model: str, minimal_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Gemini via the existing client. Expects generate_json(prompt, model=...)
    to return (text, usage).
    """
    prompt = SYSTEM_AND_TASK_PROMPT + json.dumps(minimal_case, ensure_ascii=False, indent=2)

    text, usage = await client.generate_json(prompt, model=model)
    _ = usage  # you can log or track if desired

    parsed = extract_first_json_object(text)

    # Minimal schema validation
    required_keys = ["class", "direction", "chosen", "gold_best", "reason", "confidence"]
    for key in required_keys:
        if key not in parsed:
            raise ValueError(f"Missing key '{key}' in LLM response: {parsed}")

    return parsed


async def async_main(args):
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    model_name = args.model or getattr(config, "GEMINI_SELECTOR_MODEL_NAME", None)
    if not model_name:
        raise ValueError("No Gemini model name provided (via --model or config.GEMINI_SELECTOR_MODEL_NAME).")

    client = GeminiClient(api_key=config.GEMINI_API_KEY)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        cases = data.get("cases") or data.get("items") or data.get("data") or []
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError("Input JSON must be a list of cases or a dict containing such a list.")

    logger.info(f"Loaded {len(cases)} raw cases from {args.input}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, case in enumerate(cases, start=1):
            minimal_case = build_minimal_case(case)

            if not minimal_case.get("query") or not minimal_case.get("chosen_term"):
                logger.warning(f"Skipping case {i}: missing query or chosen_term")
                continue

            if not minimal_case["ground_truth_terms"]:
                logger.warning(f"Skipping case {i}: no ground_truth_terms")
                continue

            try:
                annotation = await call_gemini(client, model_name, minimal_case)
            except Exception as e:
                logger.error(f"Failed on case {i}: {e}")
                continue

            record = {
                "index": i,
                "minimal_case": minimal_case,
                "annotation": annotation,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"Wrote {written} annotated cases to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Annotate ontology drift types with Gemini.")
    parser.add_argument(
        "--input",
        type=str,
        default="cafeteria_results/formated_results/cafeteria_misses_gemini_klex15_ksem15_tau0.4.json",
        help="Path to input JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cafeteria_results/formated_results/cafeteria_misses_drift_annotations_gemini.jsonl",
        help="Path to output JSONL file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (overrides config.GEMINI_SELECTOR_MODEL_NAME if provided)."
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
