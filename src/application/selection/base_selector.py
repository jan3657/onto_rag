# application/selection/base_selector.py
import logging
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from src.domain.ports import Retriever
from src import config
from src.utils.token_tracker import token_tracker

logger = logging.getLogger(__name__)

class BaseSelector(ABC):
    """
    Abstract base class for LLM-based term selectors.

    This class provides the common framework for loading
    prompts, formatting
    candidate lists, and parsing the final LLM response. Subclasses must
    implement the provider-specific `__init__` setup and the `_call_llm` method.
    """

    def __init__(self, retriever: Retriever, model_name: str):
        """
        Initializes the BaseSelector.

        Args:
            retriever (Retriever): An initialized retriever instance.
            model_name (str): The name of the LLM model to use.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.prompt_template = self._load_prompt_template()
        # Store the most recent prompt and raw LLM response for debugging/UX purposes
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        logger.info(f"{self.__class__.__name__} initialized for model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from the file."""
        template_path = config.SELECTOR_PROMPT_TEMPLATE_PATH
        try:
            with template_path.open('r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found at {template_path}")
            raise

    def _format_candidates_for_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """Formats the list of candidate documents into a string for the prompt."""
        formatted_list = []
        for i, candidate in enumerate(candidates):
            term_id = candidate.get("id")
            if not term_id:
                continue

            details = self.retriever.get_term_details(term_id)
            if not details:
                continue

            label = details.get("label", "N/A")
            definition = details.get("definition", "No definition available.")
            synonyms = "; ".join(details.get("synonyms", [])) or "None"

            formatted_list.append(
                f"{i+1}. ID: {term_id}\n"
                f"   Label: {label}\n"
                f"   Definition: {definition}\n"
                f"   Synonyms: {synonyms}"
            )
        return "\n\n".join(formatted_list)
        
    def _parse_and_validate_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parses the JSON string from the LLM and validates its structure.
        
        Args:
            response_text (str): The raw string content from the LLM, expected to be JSON.

        Returns:
            A validated dictionary or None if parsing or validation fails.
        """
        def _extract_first_json_object(text: str) -> Optional[str]:
            s = text
            # Try to pull out a fenced code block first (```...```)
            try:
                start = s.find("```")
                if start != -1:
                    end = s.find("```", start + 3)
                    if end != -1:
                        block = s[start + 3:end]
                        # Drop optional language hint like 'json' at the start of the block
                        block_stripped = block.lstrip()
                        if block_stripped.lower().startswith("json"):
                            # remove the 'json' word and an optional newline
                            block = block_stripped[4:].lstrip("\n\r ")
                        s = block
            except Exception:
                pass

            # Now try to extract the first balanced JSON object
            start_brace = s.find("{")
            while start_brace != -1:
                depth = 0
                in_string = False
                escape = False
                for i, ch in enumerate(s[start_brace:], start=start_brace):
                    if in_string:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                        continue
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                return s[start_brace:i + 1]
                # No balanced object found starting here; try the next '{'
                start_brace = s.find("{", start_brace + 1)
            return None

        def _wrap_no_selection() -> Dict[str, Any]:
            return {
                'choices': []
            }

        def _normalize_choice(raw_choice: Any) -> Optional[Dict[str, Any]]:
            if raw_choice is None:
                return None
            if isinstance(raw_choice, str):
                cid = raw_choice.strip()
                if cid in {"-1", "0", ""}:
                    return None
                return {"id": cid, "selector_explanation": "No explanation provided.", "selector_confidence": None}
            if not isinstance(raw_choice, dict):
                return None
            cid = raw_choice.get("id") or raw_choice.get("chosen_id") or raw_choice.get("candidate_id")
            if cid is None:
                return None
            cid = str(cid).strip()
            if cid in {"-1", "0", ""}:
                return None
            expl = raw_choice.get("explanation") or raw_choice.get("selector_explanation") or "No explanation provided."
            try:
                conf = raw_choice.get("confidence_score")
                conf = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf = None
            return {
                "id": cid,
                "selector_explanation": expl,
                "selector_confidence": conf,
            }

        raw = (response_text or "").strip()

        # Fast-path: handle plain numeric/string sentinel like -1 / "-1"
        if raw in {"-1", "0", '"-1"', '"0"'}:
            return _wrap_no_selection()

        # Try direct JSON parse first
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract an embedded JSON object from mixed prose
            embedded = _extract_first_json_object(raw)
            if embedded:
                try:
                    parsed = json.loads(embedded)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from LLM response (embedded object parse failed): {response_text}")
                    return None
            else:
                logger.error(f"Failed to decode JSON from LLM response: {response_text}")
                return None

        # Accept special cases where the model returned a bare number or string
        if isinstance(parsed, (int, float)):
            # Treat -1/0 as no selection; everything else is invalid
            if int(parsed) in (-1, 0):
                return _wrap_no_selection()
            logger.error(f"Invalid numeric response from LLM: {parsed}")
            return None
        if isinstance(parsed, str):
            t = parsed.strip()
            if t in ("-1", "0"):
                return _wrap_no_selection()
            logger.error(f"Invalid string response from LLM: {parsed}")
            return None

        choices: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            if "choices" in parsed and isinstance(parsed["choices"], list):
                for ch in parsed["choices"]:
                    norm = _normalize_choice(ch)
                    if norm:
                        choices.append(norm)
            elif "chosen_ids" in parsed and isinstance(parsed["chosen_ids"], list):
                for ch in parsed["chosen_ids"]:
                    norm = _normalize_choice(ch)
                    if norm:
                        choices.append(norm)
            elif "chosen_id" in parsed:
                norm = _normalize_choice(parsed.get("chosen_id"))
                if norm:
                    # carry explanation if present at top level
                    if parsed.get("explanation"):
                        norm["selector_explanation"] = parsed["explanation"]
                    choices.append(norm)
        elif isinstance(parsed, list):
            for ch in parsed:
                norm = _normalize_choice(ch)
                if norm:
                    choices.append(norm)
        else:
            logger.error(f"Invalid LLM response type (expected object or list): {type(parsed).__name__}")
            return None

        # Default choice explanations if missing
        for c in choices:
            c.setdefault("selector_explanation", "No explanation provided.")

        validated_result = {
            "choices": choices,
        }
        if choices:
            validated_result["chosen_id"] = choices[0]["id"]
        return validated_result

    @abstractmethod
    async def _call_llm(self, prompt: str, query: str) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Makes the actual API call to the specific LLM provider.

        This method must be implemented by subclasses. It should handle
        provider-specific API calls, authentication, and error handling.
 
        Args:
            prompt (str): The fully formatted prompt to send to the LLM.
            query (str): The original user query, for logging purposes.

        Returns:
            A tuple containing (response_text, token_usage_dict)
        """
        pass

    async def select_best_term(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        context: str = "",
        feedback: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Run the selector LLM to choose the best candidate.

        Parameters
        ----------
        query : str
            The query string for this iteration.
        candidates : list[dict]
            Retrieved candidate terms.
        context : str, optional
            Surrounding text window.
        feedback : str, optional
            Scorer feedback from the previous attempt to help focus selection.
        """
        if not candidates:
            return None

        candidate_str = self._format_candidates_for_prompt(candidates)
        prompt = (
            self.prompt_template
                .replace("[USER_ENTITY]", query)
                .replace("[CANDIDATE_LIST]", candidate_str)
                .replace("[CONTEXT]", context or "")
                .replace("[SCORER_FEEDBACK]", feedback or "")
        )

        logger.debug(f"Selector Prompt:\n---\n{prompt}\n---")
        self.last_prompt = prompt

        # single call to provider client
        response_text, token_usage = await self._call_llm(prompt, query)
        self.last_raw_response = response_text or ""
        
        if token_usage:
            token_tracker.record_usage(
                model_name=self.model_name,
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0),
                call_type='selector'
            )
        
        if response_text is None:
            return None
            
        logger.debug(f"Selector Raw Response:\n---\n{response_text}\n---") 
        
        # Use the centralized parsing and validation method
        return self._parse_and_validate_response(response_text)
