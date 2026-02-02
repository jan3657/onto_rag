# src/utils/response_parsing.py
"""
Utilities for parsing LLM responses, especially for reasoning/thinking models
that may output <think>...</think> or similar blocks before the actual JSON.
"""

import re
from typing import Optional


def strip_thinking_blocks(text: str) -> str:
    """
    Strip reasoning/thinking blocks from model output.
    
    Handles formats like:
    - <think>...</think>
    - <reasoning>...</reasoning>
    - Any similar XML-style thinking tags
    - Models that output just </think> without opening tag (take content after it)
    
    Returns the text after the last closing tag, or the original text
    if no thinking blocks are found.
    """
    if not text:
        return text
    
    result = text
    
    # First, handle complete tag pairs (opening and closing)
    # Pattern to match thinking blocks (greedy, matches the last closing tag)
    patterns = [
        r'<think>.*?</think>\s*',
        r'<thinking>.*?</thinking>\s*',
        r'<reasoning>.*?</reasoning>\s*',
    ]
    
    for pattern in patterns:
        # Use DOTALL to match across newlines
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # Second, handle cases where model outputs reasoning then just </think> (no opening tag)
    # e.g., "We need to evaluate... reasoning text\n</think>\n{json}"
    closing_tags = ['</think>', '</thinking>', '</reasoning>']
    for tag in closing_tags:
        tag_lower = tag.lower()
        result_lower = result.lower()
        if tag_lower in result_lower:
            # Find the last occurrence and take everything after it
            idx = result_lower.rfind(tag_lower)
            if idx != -1:
                result = result[idx + len(tag):].strip()
    
    return result.strip()


def extract_json_after_thinking(text: str) -> str:
    """
    Extract JSON content that appears after thinking blocks.
    
    This is more aggressive - it looks for the last occurrence of a JSON
    object in the response, which is where models typically put the final answer.
    """
    if not text:
        return text
    
    # First try stripping thinking blocks
    cleaned = strip_thinking_blocks(text)
    
    # If we have a clean result starting with { or [, return it
    if cleaned.startswith('{') or cleaned.startswith('['):
        return cleaned
    
    # Otherwise, try to find the last JSON object
    # Look for the last { that eventually has a matching }
    last_brace = text.rfind('{')
    if last_brace != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[last_brace:], start=last_brace):
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
                        return text[last_brace:i + 1]
    
    return cleaned


def clean_llm_json_response(text: str) -> str:
    """
    Clean an LLM response that should contain JSON.
    
    Combines multiple cleaning strategies:
    1. Strip thinking blocks
    2. Remove markdown code fences
    3. Extract JSON if buried in other text
    
    Use this as the primary method for cleaning responses before JSON parsing.
    """
    if not text:
        return text
    
    # Strip thinking blocks first
    cleaned = strip_thinking_blocks(text)
    
    # Remove markdown code fences
    cleaned = cleaned.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    
    # If it doesn't look like JSON yet, try to extract it
    if not (cleaned.startswith('{') or cleaned.startswith('[')):
        cleaned = extract_json_after_thinking(text)
        # Clean code fences again if needed
        cleaned = cleaned.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    
    return cleaned
