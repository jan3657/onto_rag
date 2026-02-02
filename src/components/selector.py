import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from src.interfaces import Retriever, LLMClient
from src.components.llm_client import GeminiClient
from src import config
from src.utils.token_tracker import token_tracker
from src.utils.tracing import trace_log
from src.utils.response_parsing import clean_llm_json_response
from src.utils.json_schemas import SELECTOR_SCHEMA

logger = logging.getLogger(__name__)

class Selector:
    """Uses an LLM to select the best ontology term from candidates."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
    ):
        self.retriever = retriever
        self.model_name = model_name or config.GEMINI_SELECTOR_MODEL_NAME
        
        if llm_client:
            self.client = llm_client
        else:
            # Fallback to Gemini for backwards compatibility
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            self.client = GeminiClient(api_key=config.GEMINI_API_KEY)
        
        self.prompt_template = self._load_prompt_template()
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        logger.info(f"Selector initialized for model: {self.model_name}")

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
        """Parses the JSON string from the LLM and validates its structure."""
        def _extract_first_json_object(text: str) -> Optional[str]:
            s = text
            try:
                start = s.find("```")
                if start != -1:
                    end = s.find("```", start + 3)
                    if end != -1:
                        block = s[start + 3:end]
                        block_stripped = block.lstrip()
                        if block_stripped.lower().startswith("json"):
                            block = block_stripped[4:].lstrip("\n\r ")
                        s = block
            except Exception:
                pass

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
                start_brace = s.find("{", start_brace + 1)
            return None

        def _wrap_no_selection() -> Dict[str, Any]:
            return {
                'chosen_id': '-1',
                'selector_explanation': 'Model returned no selection (-1).'
            }

        raw = (response_text or "").strip()
        
        # Strip thinking blocks from reasoning models like Nemotron
        raw = clean_llm_json_response(raw)

        if raw in {"-1", "0", '"-1"', '"0"'}:
            return _wrap_no_selection()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
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

        if isinstance(parsed, (int, float)):
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

        if not isinstance(parsed, dict):
            logger.error(f"Invalid LLM response type (expected object): {type(parsed).__name__}")
            return None

        if "chosen_id" not in parsed or parsed.get("chosen_id") is None:
            logger.error("LLM response is invalid: Missing the mandatory 'chosen_id' key. Response: %s", parsed)
            return None

        validated_result = {'chosen_id': str(parsed['chosen_id'])}
        if 'explanation' in parsed:
            validated_result['selector_explanation'] = parsed['explanation']
        else:
            logger.warning("LLM response missing 'explanation' key. Using default value.")
            validated_result['selector_explanation'] = 'No explanation provided.'

        return validated_result

    async def select_best_term(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        context: str = "",
        feedback: str = "",
        trace_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Run the selector LLM to choose the best candidate."""
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

        logger.debug(f"[SELECTOR_PROMPT] query='{query}' | full_prompt:\n{prompt}")
        self.last_prompt = prompt
        
        if trace_id:
            trace_log("llm_selector_prompt", trace_id, query, query, 0,
                      prompt_length=len(prompt), candidate_count=len(candidates))

        response_text, token_usage = await self.client.generate_json(
            prompt, model=self.model_name, json_schema=SELECTOR_SCHEMA
        )
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
            
        logger.debug(f"[SELECTOR_RAW_LLM_RESPONSE] query='{query}' | raw_text:\n{response_text}")
        
        parsed = self._parse_and_validate_response(response_text)
        
        if trace_id:
            if parsed:
                trace_log("llm_selector_response", trace_id, query, query, 0,
                          chosen_id=parsed.get('chosen_id'),
                          explanation=parsed.get('selector_explanation', '')[:200])
            else:
                trace_log("llm_selector_parse_error", trace_id, query, query, 0,
                          raw_response=response_text[:500],
                          error="Failed to parse LLM response")
        
        return parsed
