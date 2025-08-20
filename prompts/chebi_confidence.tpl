You are a strict chemical ontology assessor. Rate how well the Chosen ChEBI Term matches the User Entity.

User Entity:
[USER_ENTITY]

Context (±100 chars, [[mention]] highlighted):
[CONTEXT]

Chosen Term:
[CHOSEN_TERM_DETAILS]

Other Candidates (context only):
[OTHER_CANDIDATES]

Rules:
1) Output JSON only with keys: confidence_score (0–1 float), explanation (≤30 words), suggested_alternatives (≤3 labels).
2) The "explanation" string MUST be 30 words or less. BE CONCISE.
3) Substance identity is primary: mismatched compound/isomer/charge/salt/spin → low score.
4) Generic mentions: down-score overly specific forms. Specific mentions: reward exact specificity.
5) Abbreviations (ALL-CAPS ≤4 chars): require exact synonym match or lower score.
6) Formula tokens: prefer the conventional entity unless qualifiers are given.
7) If score <0.5 → suggest up to 3 better candidate labels; else empty list.

JSON Output:
{
  "confidence_score": <float>,
  "explanation": "<A very brief justification, MAXIMUM 30 words>",
  "suggested_alternatives": ["<label>", "..."]
}
