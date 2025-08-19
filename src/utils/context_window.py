# src/utils/context_window.py
from typing import Optional

def make_context_window(doc_text: str,
                        mention: str,
                        start: Optional[int] = None,
                        end: Optional[int] = None,
                        radius: int = 100) -> str:
    """
    Returns a string that shows up to `radius` chars of left/right context
    around the mention. If `start`/`end` are provided, they take precedence
    (robust for repeated mentions). Falls back to the first occurrence of
    `mention` if offsets are absent.

    Output format:
      …<LEFT>[[MENTION]]<RIGHT>…
    """
    if not doc_text:
        return ""
    if start is None or end is None:
        idx = doc_text.lower().find(mention.lower())
        if idx == -1:
            # Couldn't find – just return the first 2*radius chars around center-ish
            mid = max(0, len(doc_text)//2)
            s, e = max(0, mid-radius), min(len(doc_text), mid+radius)
            return f"{'…' if s>0 else ''}{doc_text[s:e]}{'…' if e<len(doc_text) else ''}"
        start, end = idx, idx + len(mention)

    s = max(0, start - radius)
    e = min(len(doc_text), end + radius)
    left = doc_text[s:start]
    right = doc_text[end:e]
    pre = "…" if s > 0 else ""
    post = "…" if e < len(doc_text) else ""
    return f"{pre}{left}[[{doc_text[start:end]}]]{right}{post}"
