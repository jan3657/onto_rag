# src/rag_selectors/huggingface_generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Optional

from src import config

logger = logging.getLogger(__name__)

class HuggingFaceLocalGenerator:
    """
    Singleton-style class to manage a single instance of a local Hugging Face model
    and tokenizer, ensuring it's loaded only once to conserve VRAM.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HuggingFaceLocalGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_id: str = None, model_kwargs: dict = None):
        # The __init__ will be called every time, but we only load the model once.
        if hasattr(self, 'model'):
            return
            
        logger.info("Initializing HuggingFaceLocalGenerator for the first time...")
        
        self.model_id = model_id or config.HF_SELECTOR_MODEL_ID
        self.model_kwargs = model_kwargs or config.HF_MODEL_KWARGS

        if not self.model_id:
            raise ValueError("HF_SELECTOR_MODEL_ID not configured in src/config.py")

        try:
            logger.info(f"Loading tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info(f"Loading model: {self.model_id} with kwargs: {self.model_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **self.model_kwargs
            )
            logger.info("Hugging Face model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model '{self.model_id}': {e}", exc_info=True)
            raise

    def generate(self, prompt: str, generation_kwargs: Optional[dict] = None) -> Optional[str]:
        """
        Generates text using the loaded model, applying the chat template.

        Args:
            prompt (str): The user content for the prompt.
            generation_kwargs (dict, optional): Overrides for generation parameters.

        Returns:
            The generated text string, or None on failure.
        """
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not loaded. Cannot generate.")
            return None

        # The arcee model uses a chat template, which is the standard way to prompt it.
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply the template and move to the model's device
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            attention_mask = torch.ones_like(input_ids)

            gen_kwargs = config.HF_GENERATION_KWARGS.copy()
            if generation_kwargs:
                gen_kwargs.update(generation_kwargs)
            
            # 3. Pass both tensors as explicit keyword arguments to generate()
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, # This solves the warning
                **gen_kwargs
            )
            
            # Decode the response, skipping the prompt part
            # outputs[0] contains the full sequence (prompt + response)
            # input_ids.shape[-1] gives us the length of the prompt
            response_ids = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()

        except Exception as e:
            logger.error(f"Error during Hugging Face model generation: {e}", exc_info=True)
            return None