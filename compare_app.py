# app.py — Streamlit app to compare two ontology-mapping outputs side-by-side
# Author: ChatGPT
#
# Usage (suggested):
#   uv venv
#   uv pip install streamlit pandas
#   streamlit run app.py
#
# Inputs:
#   • mapped_ingredients_foodsem.json
#   • mapped_ingredients_ontorag.json
#   • ontology_dump_foodon.json (plus any other ontologies if needed; unknown CURIEs are handled gracefully)
#
# Notes:
#   • We do NOT parse LLM free-text fields. We rely on the structured `mapped_ingredients` list.
#   • We align rows by the normalized `original_ingredient` string (casefolded, punctuation-stripped).
#   • If an approach has no match for an ingredient, its box is left empty.
#   • You can choose the better match per ingredient and export your decisions.

from __future__ import annotations
import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Ontology Mapping Comparator",
    layout="wide",
)

st.title("🔍 Ontology Mapping Comparator")
st.write(
    "Compare mappings from two approaches (FoodSem vs. OntoRAG) for each original entity, "
    "pulling labels/definitions from an ontology dump. Then pick the better match."
)

# -----------------------------
# Helpers
# -----------------------------
PUNCT_RE = re.compile(r"[\s\u200b\-–—,_;:()\[\]{}]+")


def normalize_term(s: str) -> str:
    """Normalize an original ingredient string for stable matching across approaches.
    - lowercases
    - strips punctuation-like separators to single spaces
    - collapses whitespace
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_purl(curie_or_url: Optional[str]) -> Optional[str]:
    """Return an OBO PURL for a CURIE like 'FOODON:03301116'.
    If it looks like a URL already, return it as-is. Unknown formats return None.
    """
    if not curie_or_url:
        return None
    s = curie_or_url.strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    # Drop optional 'obo:' prefix if present
    s = s[4:] if s.lower().startswith("obo:") else s
    m = re.match(r"^([A-Za-z][A-Za-z0-9]+):([A-Za-z0-9_]+)$", s)
    if not m:
        return None
    prefix, local = m.groups()
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"


@dataclass
class OntInfo:
    id: Optional[str]
    label: Optional[str]
    synonyms: List[str]
    definition: Optional[str]


def find_ont_info(ontologies: Dict[str, Dict], curie: Optional[str]) -> OntInfo:
    """Look up ontology info by CURIE in the loaded ontology dump(s).
    The `ontologies` dict is expected to be a single merged dict keyed by CURIE.
    We try a few variants to be forgiving (obo: prefix, underscores in localID).
    """
    if not curie:
        return OntInfo(None, None, [], None)

    c = curie.strip()
    # Try exact
    if c in ontologies:
        ent = ontologies[c]
        return OntInfo(curie, ent.get("label"), ent.get("synonyms", []) or [], ent.get("definition"))

    # Try removing 'obo:' prefix
    if c.lower().startswith("obo:") and c[4:] in ontologies:
        ent = ontologies[c[4:]]
        return OntInfo(curie, ent.get("label"), ent.get("synonyms", []) or [], ent.get("definition"))

    # Sometimes IDs like CHEBI:00004033 vary in zero-padding; try unpadded
    m = re.match(r"^([A-Za-z][A-Za-z0-9]+):(0*)([0-9]+)$", c)
    if m:
        prefix, zeros, num = m.groups()
        unpadded = f"{prefix}:{int(num)}"
        if unpadded in ontologies:
            ent = ontologies[unpadded]
            return OntInfo(curie, ent.get("label"), ent.get("synonyms", []) or [], ent.get("definition"))

    # Try underscore variant (rare in dumps, common in PURLs)
    underscore_variant = re.sub(":", "_", c)
    if underscore_variant in ontologies:
        ent = ontologies[underscore_variant]
        return OntInfo(curie, ent.get("label"), ent.get("synonyms", []) or [], ent.get("definition"))

    return OntInfo(curie, None, [], None)


@dataclass
class Row:
    entry_id: str
    original_raw: str
    original_norm: str
    foodsem_id: Optional[str]
    ontorag_id: Optional[str]


def extract_rows_for_entry(entry_id: str,
                           foodsem_entry: Optional[dict],
                           ontorag_entry: Optional[dict]) -> List[Row]:
    """Build a union of original ingredients for a given entry id and attach IDs from both approaches."""
    # Map normalized original -> (raw, id)
    def build_side_map(entry: Optional[dict]) -> Dict[str, Tuple[str, Optional[str]]]:
        out: Dict[str, Tuple[str, Optional[str]]] = {}
        if not entry or not isinstance(entry, dict):
            return out

        items = entry.get("mapped_ingredients", []) or []
        # Coerce unexpected shapes into a list of dicts
        if isinstance(items, str):
            # Try to parse if it's a JSON-encoded string; otherwise ignore
            try:
                parsed = json.loads(items)
                items = parsed
            except Exception:
                items = []
        if isinstance(items, dict):
            # Some producers may provide a dict keyed by original ingredient
            items = list(items.values())
        if not isinstance(items, list):
            items = []

        for mi in items:
            if not isinstance(mi, dict):
                continue
            raw = (mi.get("original_ingredient") or "").strip()
            norm = normalize_term(raw)
            mres = mi.get("mapping_result") or {}
            if not isinstance(mres, dict):
                mres = {}
            mid = mres.get("id")
            # take the first non-null mapping per normalized original
            if norm not in out or (out[norm][1] is None and mid):
                out[norm] = (raw, mid)
        return out

    f_map = build_side_map(foodsem_entry)
    o_map = build_side_map(ontorag_entry)

    # Union of all originals
    all_norms = list({*f_map.keys(), *o_map.keys()})
    all_norms.sort()

    rows: List[Row] = []
    for norm in all_norms:
        f_raw, f_id = f_map.get(norm, (None, None))
        o_raw, o_id = o_map.get(norm, (None, None))
        # Prefer a non-empty raw for display
        raw = f_raw or o_raw or norm
        rows.append(Row(
            entry_id=entry_id,
            original_raw=raw,
            original_norm=norm,
            foodsem_id=f_id,
            ontorag_id=o_id,
        ))
    return rows


@st.cache_data(show_spinner=False)
def load_json_filelike(filelike) -> dict:
    return json.load(filelike)


@st.cache_data(show_spinner=False)
def load_json_path(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Sidebar: inputs
# -----------------------------
st.sidebar.header("📥 Inputs")

col_path, col_upload = st.sidebar.columns(2)
with col_path:
    path_foodsem = st.text_input("FoodSem JSON path (optional)")
    path_ontorag = st.text_input("OntoRAG JSON path (optional)")
    path_ontology = st.text_input("Ontology dump JSON path (optional)")
with col_upload:
    up_foodsem = st.file_uploader("Or upload FoodSem JSON", type=["json"], key="fs")
    up_ontorag = st.file_uploader("Or upload OntoRAG JSON", type=["json"], key="or")
    up_ontology = st.file_uploader("Or upload Ontology dump JSON", type=["json"], key="od")

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("🛠 Debug mode", value=True)
if st.sidebar.button("Clear cache & rerun"):
    st.cache_data.clear()
    st.rerun()

# Load data (path has priority if both given)
foodsem = None
ontorag = None
ontology = None

try:
    if path_foodsem:
        foodsem = load_json_path(path_foodsem)
    elif up_foodsem is not None:
        foodsem = load_json_filelike(up_foodsem)
except Exception as e:
    st.sidebar.error(f"Failed to load FoodSem JSON: {e}")
    if debug_mode:
        st.sidebar.exception(e)

try:
    if path_ontorag:
        ontorag = load_json_path(path_ontorag)
    elif up_ontorag is not None:
        ontorag = load_json_filelike(up_ontorag)
except Exception as e:
    st.sidebar.error(f"Failed to load OntoRAG JSON: {e}")
    if debug_mode:
        st.sidebar.exception(e)

try:
    if path_ontology:
        ontology = load_json_path(path_ontology)
    elif up_ontology is not None:
        ontology = load_json_filelike(up_ontology)
except Exception as e:
    st.sidebar.error(f"Failed to load Ontology dump JSON: {e}")
    if debug_mode:
        st.sidebar.exception(e)

if ontology is None:
    st.info("Upload or provide a path to the ontology dump JSON to see labels/definitions.")
    ontology = {}

if not foodsem and not ontorag:
    st.warning("Please load at least one mapping file (FoodSem and/or OntoRAG).")
    if debug_mode:
        st.write({"cwd": __import__("os").getcwd()})
    st.stop()

# Ensure dicts (if a list sneaks in, convert index -> entry)
if isinstance(foodsem, list):
    foodsem = {str(i): v for i, v in enumerate(foodsem)}
if isinstance(ontorag, list):
    ontorag = {str(i): v for i, v in enumerate(ontorag)}
foodsem = foodsem or {}
ontorag = ontorag or {}

# Collect entry IDs (union across both files)
entry_ids = sorted(set(foodsem.keys()) | set(ontorag.keys()))

st.sidebar.markdown("---")
if not entry_ids:
    st.error("No entry IDs found in the loaded JSON(s). Enable Debug mode for details.")
    if debug_mode:
        st.write({
            "foodsem_type": type(foodsem).__name__,
            "ontorag_type": type(ontorag).__name__,
            "foodsem_top_level_keys": list(foodsem.keys())[:5],
            "ontorag_top_level_keys": list(ontorag.keys())[:5],
        })
    st.stop()

chosen_entry = st.sidebar.selectbox(
    "Choose an entry (product/document ID)", entry_ids, index=0
)

show_only_unresolved = st.sidebar.checkbox("Show only unresolved ingredients", value=False)
search_filter = st.sidebar.text_input("Search original ingredient (contains)")

st.sidebar.markdown("---")
keep_only_with_both = st.sidebar.checkbox(
    "Show only rows where BOTH approaches have a mapping", value=False
)

# Session state to keep decisions across reruns
if "decisions" not in st.session_state:
    # key: (entry_id, original_norm) -> { choice: "foodsem"|"ontorag"|"skip", selected_id: Optional[str], note: str }
    st.session_state.decisions = {}

# Build rows for the selected entry
rows = extract_rows_for_entry(
    chosen_entry,
    foodsem.get(chosen_entry),
    ontorag.get(chosen_entry),
)

if debug_mode:
    st.write({
        "selected_entry": chosen_entry,
        "foodsem_has_entry": chosen_entry in foodsem,
        "ontorag_has_entry": chosen_entry in ontorag,
        "foodsem_mapped_count": len((foodsem.get(chosen_entry) or {}).get("mapped_ingredients", []) or []),
        "ontorag_mapped_count": len((ontorag.get(chosen_entry) or {}).get("mapped_ingredients", []) or []),
        "union_rows": len(rows),
    })

if search_filter:
    nf = search_filter.lower().strip()
    rows = [r for r in rows if nf in r.original_raw.lower()]

if keep_only_with_both:
    rows = [r for r in rows if r.foodsem_id and r.ontorag_id]

# Display original ingredient string (if present)
original_text = (
    (foodsem.get(chosen_entry) or {}).get("original_ingredients")
    or (ontorag.get(chosen_entry) or {}).get("original_ingredients")
)
if original_text:
    with st.expander("Original ingredients text", expanded=False):
        st.write(original_text)
else:
    if debug_mode:
        st.info("No 'original_ingredients' text found for this entry.")

# -----------------------------
# Table rendering helpers
# -----------------------------

def card_for(curie: Optional[str]):
    """Render a small info card for the given CURIE using ontology dump."""
    info = find_ont_info(ontology, curie)
    purl = to_purl(curie) if curie else None
    if not curie:
        st.write("—")
        return

    st.markdown(f"**{info.label or 'Unknown label'}**  ")
    st.caption(curie)
    if purl:
        st.markdown(f"[Open in OBO PURL]({purl})")
    if info.definition:
        with st.popover("Definition"):
            st.write(info.definition)
    if info.synonyms:
        with st.popover("Synonyms"):
            st.write(", ".join(info.synonyms))


# -----------------------------
# Debug info
# -----------------------------
if debug_mode:
    st.sidebar.markdown("### Debug snapshot")
    st.sidebar.write({
        "foodsem_loaded": bool(foodsem),
        "ontorag_loaded": bool(ontorag),
        "ontology_loaded": bool(ontology),
        "foodsem_entries": len(foodsem) if isinstance(foodsem, dict) else None,
        "ontorag_entries": len(ontorag) if isinstance(ontorag, dict) else None,
    })

# -----------------------------
# Main grid
# -----------------------------
left, mid, right = st.columns([2, 3, 3])
with left:
    st.subheader("Original")
with mid:
    st.subheader("FoodSem match")
with right:
    st.subheader("OntoRAG match")

resolved_count = 0
for r in rows:
    # Filter unresolved if requested (we do it here to account for prior decisions)
    key = (r.entry_id, r.original_norm)
    decision = st.session_state.decisions.get(key, {"choice": None, "selected_id": None, "note": ""})

    if show_only_unresolved and decision.get("choice") in {"foodsem", "ontorag", "skip"}:
        continue

    c1, c2, c3 = st.columns([2, 3, 3])
    with c1:
        st.markdown(f"**{r.original_raw}**")

    with c2:
        card_for(r.foodsem_id)

    with c3:
        card_for(r.ontorag_id)

    # Choice row
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 3])

    options = []
    if r.foodsem_id:
        options.append("FoodSem")
    if r.ontorag_id:
        options.append("OntoRAG")
    options.append("Skip")

    default_idx = None
    if decision.get("choice"):
        try:
            default_idx = options.index(decision["choice"].capitalize())
        except Exception:
            default_idx = None

    with cc1:
        choice = st.radio(
            "Pick best",
            options,
            index=default_idx if default_idx is not None else (0 if options else None),
            horizontal=True,
            key=f"choice::{r.entry_id}::{r.original_norm}",
        )

    selected_id = None
    if choice == "FoodSem":
        selected_id = r.foodsem_id
    elif choice == "OntoRAG":
        selected_id = r.ontorag_id

    with cc2:
        st.caption("Selected ID")
        st.code(selected_id or "—", language="text")

    with cc3:
        note_key = f"note::{r.entry_id}::{r.original_norm}"
        note_val = st.text_input("Notes (optional)", value=decision.get("note", ""), key=note_key)

    # Persist
    st.session_state.decisions[key] = {
        "choice": choice.lower(),
        "selected_id": selected_id,
        "note": st.session_state.get(note_key, ""),
        "foodsem_id": r.foodsem_id,
        "ontorag_id": r.ontorag_id,
        "original_raw": r.original_raw,
    }

    if choice.lower() in {"foodsem", "ontorag", "skip"}:
        resolved_count += 1

    # Visual separator between entities for readability
    try:
        st.divider()
    except Exception:
        # Fallback for older Streamlit versions
        st.markdown("---")

st.markdown("---")
st.success(f"Resolved {resolved_count} / {len(rows)} rows shown.")

# -----------------------------
# Export decisions
# -----------------------------
export_rows = []
for (entry_id, original_norm), d in st.session_state.decisions.items():
    export_rows.append({
        "entry_id": entry_id,
        "original_norm": original_norm,
        "original": d.get("original_raw"),
        "choice": d.get("choice"),
        "selected_id": d.get("selected_id"),
        "foodsem_id": d.get("foodsem_id"),
        "ontorag_id": d.get("ontorag_id"),
        "note": d.get("note", ""),
    })

if export_rows:
    df = pd.DataFrame(export_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download decisions (CSV)",
        data=csv_bytes,
        file_name="ontology_mapping_decisions.csv",
        mime="text/csv",
    )

    json_bytes = json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download decisions (JSON)",
        data=json_bytes,
        file_name="ontology_mapping_decisions.json",
        mime="application/json",
    )
else:
    st.info("Make some selections to enable export.")
