import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TokenTracker:
    """
    A singleton class to track LLM token usage across the application.
    This helps in monitoring costs and performance.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenTracker, cls).__new__(cls)
            # Initialize state only once
            cls._instance.counts = defaultdict(lambda: defaultdict(int))
            cls._instance.call_counts = defaultdict(lambda: defaultdict(int))
        return cls._instance

    def record_usage(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        call_type: str,
    ):
        """
        Records the token usage for a specific LLM call.

        Args:
            model_name (str): The name of the model used.
            prompt_tokens (int): The number of tokens in the prompt.
            completion_tokens (int): The number of tokens in the completion.
            call_type (str): The type of call (e.g., 'selector', 'scorer').
        """
        if prompt_tokens is None or completion_tokens is None:
            return 
            
        self.counts[model_name]['prompt_tokens'] += prompt_tokens
        self.counts[model_name]['completion_tokens'] += completion_tokens
        self.counts[model_name]['total_tokens'] += prompt_tokens + completion_tokens

        self.call_counts[model_name][call_type] += 1

        logger.debug(
            f"Token usage recorded for {model_name} ({call_type}): "
            f"Prompt={prompt_tokens}, Completion={completion_tokens}, "
            f"Total={prompt_tokens + completion_tokens}"
        )

    def get_total_usage(self) -> dict:
        """Returns the aggregated token counts for all models."""
        total = defaultdict(int)
        for model_data in self.counts.values():
            total['prompt_tokens'] += model_data['prompt_tokens']
            total['completion_tokens'] += model_data['completion_tokens']
            total['total_tokens'] += model_data['total_tokens']
        return dict(total)

    def report_usage(self) -> str:
        """
        Generates a formatted string report of the token usage.
        """
        if not self.counts:
            return "--- Token Usage Report ---\nNo LLM calls with token usage data were made.\n"

        report_lines = ["\n--- Token Usage Report ---"]
        total_usage = self.get_total_usage()

        report_lines.append(
            f"Overall Total: {total_usage.get('total_tokens', 0):,} tokens "
            f"(Prompt: {total_usage.get('prompt_tokens', 0):,}, "
            f"Completion: {total_usage.get('completion_tokens', 0):,})"
        )
        report_lines.append("-" * 26)

        for model, data in self.counts.items():
            report_lines.append(f"Model: {model}")
            report_lines.append(
                f"  - Total:      {data['total_tokens']:,} tokens"
            )
            report_lines.append(
                f"  - Prompt:     {data['prompt_tokens']:,} tokens"
            )
            report_lines.append(
                f"  - Completion: {data['completion_tokens']:,} tokens"
            )
            
            call_breakdown = []
            if model in self.call_counts:
                for call_type, count in self.call_counts[model].items():
                    call_breakdown.append(f"{call_type}: {count}")
            if call_breakdown:
                 report_lines.append(f"  - API Calls:  {', '.join(call_breakdown)}")

        report_lines.append("--------------------------\n")

        return "\n".join(report_lines)

# Create a singleton instance for global use
token_tracker = TokenTracker()
