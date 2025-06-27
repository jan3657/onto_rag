# src/selectors/ollama_selector.py
import os
import logging
import json
from typing import List, Dict, Any, Optional

import ollama # <-- New import

from src.retriever.hybrid_retriever import HybridRetriever
from src import config

logger = logging.getLogger(__name__)

class OllamaSelector:
    """
    Uses a local LLM via Ollama to select the best ontology term
    from a list of candidates.
    """
    def __init__(self, retriever: HybridRetriever):
        """
        Initializes the OllamaSelector.

        Args:
            retriever (HybridRetriever): An initialized retriever instance,
                                         used to fetch full details of candidate terms.
        """
        self.retriever = retriever
        self.prompt_template = self._load_prompt_template()
        
        # We get the model name from config, but no API key or client is needed for Ollama.
        self.model_name = config.OLLAMA_SELECTOR_MODEL_NAME # Assumes you have this in your config
        
        # You might want to add a check here to ensure the Ollama service is running.
        try:
            ollama.ps()
            logger.info("Ollama service is running. Selector initialized for model: %s", self.model_name)
        except Exception as exc:
            logger.error("Ollama service not detected. Please ensure Ollama is running.")
            raise ConnectionError("Ollama service not available.") from exc


    def _load_prompt_template(self) -> str:
        """Loads the prompt template from the file."""
        # This method is unchanged
        template_path = os.path.join(config.PROJECT_ROOT, "prompts", "strict_final_selection.tpl")
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("Prompt template not found at %s", template_path)
            raise

    def _format_candidates_for_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """Formats the list of candidate documents into a string for the prompt."""
        # This method is unchanged
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

    def select_best_term(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Prompts Ollama to select the best term and parses the JSON response.

        Args:
            query (str): The original user query.
            candidates (List[Dict[str, Any]]): The list of candidate documents.

        Returns:
            A dictionary with {"chosen_id": str, "explanation": str}, or None on failure.
        """
        if not candidates:
            return None

        candidate_str = self._format_candidates_for_prompt(candidates)
        # The full prompt is created just like before.
        full_prompt = self.prompt_template.replace("[USER_ENTITY]", query).replace("[CANDIDATE_LIST]", candidate_str)
        
        logger.info("Sending request to Ollama for query: '%s' with model '%s'", query, self.model_name)
        try:
            # --- CORRECTED OLLAMA CALL ---
            # The entire prompt goes into a single 'user' message.
            # We use format='json' to ensure the output is valid JSON.
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': full_prompt,
                    },
                ],
                format='json' # This is a key feature to get structured output!
            )

            # The response content should be a JSON string.
            response_content = response['message']['content']
            
            # Parse the JSON response
            result = json.loads(response_content)
            
            # --- MODIFIED VALIDATION LOGIC ---
            
            # 1. The 'chosen_id' key is mandatory. Fail if it's missing or null.
            if "chosen_id" not in result or result.get("chosen_id") is None:
                logger.error(
                    "LLM response is invalid: Missing the mandatory 'chosen_id' key. Response: %s",
                    result
                )
                return None
            
            # 2. Start building the result with the mandatory key.
            validated_result = {
                'chosen_id': result['chosen_id']
            }

            # 3. Handle optional 'explanation' with a specific warning if missing.
            if 'explanation' in result:
                validated_result['explanation'] = result['explanation']
            else:
                logger.warning("LLM response missing 'explanation' key. Using default value.")
                validated_result['explanation'] = 'No explanation provided.'

            # 4. Handle optional 'confidence' with distinct warnings.
            if 'confidence_score' in result:
                try:
                    # Key exists, so try to convert it
                    validated_result['confidence_score'] = float(result['confidence_score'])
                except (ValueError, TypeError):
                    # Key exists, but the value is not a valid float
                    logger.warning(
                        "Invalid confidence_score value in response: '%s'. Defaulting to 0.0.",
                        result.get('confidence_score')
                    )
                    validated_result['confidence_score'] = 0.0
            else:
                # The 'confidence_score' key itself is missing
                logger.warning("LLM response missing 'confidence_score' key. Defaulting to 0.0.")
                validated_result['confidence_score'] = 0.0

            return validated_result
            # --- END OF MODIFIED VALIDATION LOGIC ---
                
        except json.JSONDecodeError:
            # This is less likely with format='json' but is good practice to keep.
            logger.error("Failed to decode JSON from Ollama response: %s", response_content)
            return None
        except ollama.ResponseError as e:
            logger.error("An error occurred with the Ollama API call: %s - %s", e.status_code, e.error)
            return None
        except (ConnectionError, TimeoutError) as e:
            logger.error("Network error during the Ollama call: %s", e, exc_info=True)
            return None
        except RuntimeError as e:
            logger.error("Runtime error during the Ollama call: %s", e, exc_info=True)
            return None