# src/selectors/gemini_selector.py
import os
import logging
import json
from typing import List, Dict, Any, Optional

from google import genai
from google.api_core import exceptions  # <-- Import for better error handling

from src.retriever.hybrid_retriever import HybridRetriever
from src import config

logger = logging.getLogger(__name__)

class GeminiSelector:
    """
    Uses the Google Gemini model to select the best ontology term
    from a list of candidates.
    """
    def __init__(self, retriever: HybridRetriever):
        """
        Initializes the GeminiSelector.

        Args:
            retriever (HybridRetriever): An initialized retriever instance,
                                         used to fetch full details of candidate terms.
        """
        self.retriever = retriever
        self.prompt_template = self._load_prompt_template()

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        # --- CORRECTED PART 1: Client Instantiation ---
        # Instead of genai.configure(), we instantiate a client.
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model_name = config.LLM_SELECTOR_MODEL_NAME
        
        logger.info(f"GeminiSelector initialized for model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from the file."""
        template_path = os.path.join(config.PROJECT_ROOT, "prompts", "final_selection.tpl")
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
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
            
            # Fetch full details using the retriever
            details = self.retriever.get_term_details(term_id)
            if not details:
                continue

            # Format the details for display
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

    def select_best_term(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """
        Prompts Gemini to select the best term and parses the JSON response.

        Args:
            query (str): The original user query.
            candidates (List[Dict[str, Any]]): The list of candidate documents.

        Returns:
            A dictionary with {"chosen_id": str, "explanation": str}, or None on failure.
        """
        if not candidates:
            return None

        candidate_str = self._format_candidates_for_prompt(candidates)
        prompt = self.prompt_template.replace("[USER_ENTITY]", query).replace("[CANDIDATE_LIST]", candidate_str)
        
        logger.info(f"Sending request to Gemini for query: '{query}'")
        try:
            # --- CORRECTED PART 2: The API Call ---
            # Call generate_content on the client.models service.
            # Pass the prompt string to the 'contents' parameter.
            generation_config = {'temperature': 0}
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )

            # The rest of your logic is good.
            # Safety feedback check from your example code is a good practice to add here if needed.
            feedback = getattr(response, 'prompt_feedback', None)
            if feedback and any(r.blocked for r in feedback.safety_ratings or []):
                logger.warning(f"Request for query '{query}' was blocked by safety filters.")
                return None

            # Clean up the response text to extract the JSON part
            cleaned_response = response.text.strip().lstrip("```json").rstrip("```").strip()
            
            # Parse the JSON response
            result = json.loads(cleaned_response)
            if "chosen_id" in result and "explanation" in result:
                return result
            else:
                logger.error(f"LLM response is valid JSON but missing required keys: {result}")
                return None
                
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response.text}")
            return None
        except exceptions.GoogleAPIError as e:  # <-- Specific API error handling
            logger.error(f"A Google API error occurred with the Gemini call: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred with the Gemini API call: {e}", exc_info=True)
            return None