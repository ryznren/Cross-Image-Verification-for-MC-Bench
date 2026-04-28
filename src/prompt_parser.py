"""
Prompt parser for extracting structured information from text descriptions.
"""
from typing import List, Dict, Any
from PIL import Image
import json
import re


class PromptParser:
    """Parse text prompts into structured information using VLM."""

    PARSER_PROMPT = """Analyze the two images and text description to extract structured information.

Text: {text}

Output JSON format:
{{
  "target_images": [1] or [2] or [1,2] or [],
  "base_noun": "core object type",
  "attributes": ["attribute1", "attribute2"],
  "relations": ["comparison or relation"],
  "reasoning_type": "referring" or "comparison" or "reasoning"
}}

Guidelines:
- target_images: which image(s) likely contain the target (1=first, 2=second, []=none)
- base_noun: simplified noun phrase (e.g., "car", "person", "cup")
- attributes: visual properties (color, size, position)
- relations: comparative or relational clues (e.g., "larger", "darker")
- reasoning_type: classify the query type

Output only valid JSON, no other text."""

    def __init__(self, vlm):
        """
        Args:
            vlm: VLM inference instance (Qwen2VLLM)
        """
        self.vlm = vlm

    def parse(self, images: List[Image.Image], text: str) -> Dict[str, Any]:
        """
        Parse prompt into structured information.

        Args:
            images: List of 2 PIL Images
            text: Text description

        Returns:
            Parsed dict with keys: target_images, base_noun, attributes, relations, reasoning_type
        """
        prompt = self.PARSER_PROMPT.format(text=text)

        # Get VLM response
        response = self.vlm.generate_text(images, prompt)

        # Parse JSON from response
        parsed = self._extract_json(response)

        # Validate and set defaults
        if parsed is None:
            parsed = self._get_default_parse(text)

        return parsed

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from VLM response."""
        try:
            # Try to find JSON block
            json_match = re.search(r'\{[^}]*"target_images"[^}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Validate required fields
                if all(k in data for k in ["target_images", "base_noun", "reasoning_type"]):
                    return data
        except:
            pass
        return None

    def _get_default_parse(self, text: str) -> Dict[str, Any]:
        """Return default parse when VLM fails."""
        # Extract first noun as base_noun
        words = text.lower().split()
        base_noun = words[0] if words else "object"

        return {
            "target_images": [1, 2],  # Default to both images
            "base_noun": base_noun,
            "attributes": [],
            "relations": [],
            "reasoning_type": "referring"
        }
