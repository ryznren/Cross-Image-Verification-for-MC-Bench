"""
Two-stage baseline agent: Prompt Parsing + VLM Candidate Grounding.

Uses a single VLM (Qwen2.5-VL) for both stages:
  Stage 1: Parse prompt → structured info (target images, base noun, attributes)
  Stage 2: VLM grounding → generate candidate bboxes per target image
"""
from typing import List, Dict, Any
from PIL import Image


GROUNDING_PROMPT = """Look at this image carefully.

Text description: {text}

Find ALL objects matching this description. For each match, output a bounding box.

Output JSON format:
{{
  "candidates": [
    {{"bbox": [x1, y1, x2, y2], "score": 0.0-1.0}},
    ...
  ]
}}

Rules:
- bbox is [x1, y1, x2, y2] in pixel coordinates (top-left, bottom-right)
- score is your confidence that this object matches the description
- Output up to 3 candidates, sorted by score descending
- If nothing matches, output: {{"candidates": []}}
- Output only valid JSON, no other text."""


class TwoStageBaselineAgent:
    """Two-stage agent: VLM parsing + VLM grounding (no verification)."""

    def __init__(self, vlm, config: Dict[str, Any]):
        self.vlm = vlm
        self.config = config

        from src.prompt_parser import PromptParser
        self.parser = PromptParser(vlm)

    def predict(self, sample: Dict[str, Any], images: List[Image.Image]) -> Dict[str, Any]:
        """Run two-stage prediction on a sample."""
        text = sample["text"]
        sample_id = sample["sample_id"]

        # Stage 1: Parse prompt
        parsed = self.parser.parse(images, text)

        # Negative sample — no target expected
        if not parsed["target_images"]:
            return self._empty_result(sample_id)

        # Stage 2: VLM grounding on each target image
        candidates = self._ground_candidates(images, parsed, sample)

        if not candidates:
            return self._empty_result(sample_id)

        best = max(candidates, key=lambda c: c["score"])
        return {
            "sample_id": sample_id,
            "bbox": best["bbox"],
            "image_id": best["image_id"],
            "confidence": best["score"],
            "method": "two_stage_baseline",
            "parsed": parsed,
        }

    def _ground_candidates(
        self, images: List[Image.Image], parsed: Dict[str, Any], sample: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run VLM grounding on each target image."""
        import json, re

        target_indices = [idx - 1 for idx in parsed["target_images"]]
        text = sample["text"]
        all_candidates = []

        for img_idx in target_indices:
            if img_idx < 0 or img_idx >= len(images):
                continue

            image = images[img_idx]
            image_id = sample["images"][img_idx]["id"]

            prompt = GROUNDING_PROMPT.format(text=text)

            # Single-image grounding call
            response = self.vlm.generate_text([image], prompt)

            # Parse candidates from response
            boxes = self._parse_candidates(response)

            for box_info in boxes:
                bbox = box_info["bbox"]
                score = box_info.get("score", 0.5)

                # Convert [x1,y1,x2,y2] → [x,y,w,h]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if x2 > x1 and y2 > y1:
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        continue  # Invalid box

                all_candidates.append({
                    "bbox": bbox,
                    "score": score,
                    "image_id": image_id,
                })

        return all_candidates

    def _parse_candidates(self, response: str) -> List[Dict[str, Any]]:
        """Parse candidate bboxes from VLM response."""
        import json, re

        # Extract only assistant output
        if "assistant" in response:
            response = response.split("assistant")[-1]

        # Try to find JSON with candidates array
        try:
            json_match = re.search(r'\{[^{}]*"candidates"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                candidates = data.get("candidates", [])
                result = []
                for c in candidates:
                    bbox = c.get("bbox", [])
                    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                        result.append({
                            "bbox": [float(v) for v in bbox],
                            "score": float(c.get("score", 0.5)),
                        })
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to find individual bbox arrays
        try:
            bbox_matches = re.findall(
                r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',
                response
            )
            if bbox_matches:
                return [{"bbox": [float(v) for v in m], "score": 0.5} for m in bbox_matches[:3]]
        except:
            pass

        return []

    def _empty_result(self, sample_id: int) -> Dict[str, Any]:
        return {
            "sample_id": sample_id,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "image_id": -1,
            "confidence": 0.0,
            "method": "two_stage_baseline",
        }
