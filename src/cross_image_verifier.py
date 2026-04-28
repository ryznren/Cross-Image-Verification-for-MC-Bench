"""
Cross-Image Verification module.

Two verification modes:
  Local: verify a candidate crop against the text description
  Cross-image: verify a candidate considering both images together
"""
from typing import Dict, Any, List
from PIL import Image
import json
import re


LOCAL_VERIFY_PROMPT = """Look at this cropped region from an image.

Text description: {text}

Does this cropped region contain the target described in the text?

Output JSON:
{{"valid": true or false, "score": 0.0-1.0, "reason": "brief explanation"}}

Output only valid JSON."""


CROSS_IMAGE_VERIFY_PROMPT = """You are shown two full images and a cropped candidate region.
The candidate was found in image {img_num}.

Text description: {text}

Considering BOTH images together, is this candidate the correct target?
Think about:
- Does this match the description better than alternatives in the other image?
- Is this the right image for this target?

Output JSON:
{{"valid": true or false, "score": 0.0-1.0, "reason": "brief explanation"}}

Output only valid JSON."""


class CrossImageVerifier:
    """Verifies candidate bboxes using local and cross-image context."""

    def __init__(self, vlm):
        self.vlm = vlm

    def verify_local(self, image: Image.Image, bbox: List[float], text: str) -> Dict[str, Any]:
        """
        Local verification: does this crop match the text?

        Args:
            image: Full PIL image containing the candidate
            bbox: [x, y, w, h] in pixel coordinates
            text: Text description

        Returns:
            {"valid": bool, "score": float, "reason": str}
        """
        crop = self._crop_image(image, bbox)
        if crop is None:
            return {"valid": False, "score": 0.0, "reason": "invalid crop"}

        prompt = LOCAL_VERIFY_PROMPT.format(text=text)
        response = self.vlm.generate_text([crop], prompt)
        return self._parse_verification(response)

    def verify_cross_image(
        self,
        images: List[Image.Image],
        bbox: List[float],
        img_idx: int,
        text: str,
    ) -> Dict[str, Any]:
        """
        Cross-image verification: is this candidate correct considering both images?

        Args:
            images: Both full images [image1, image2]
            bbox: [x, y, w, h] in pixel coordinates
            img_idx: Which image the candidate is from (0 or 1)
            text: Text description

        Returns:
            {"valid": bool, "score": float, "reason": str}
        """
        crop = self._crop_image(images[img_idx], bbox)
        if crop is None:
            return {"valid": False, "score": 0.0, "reason": "invalid crop"}

        prompt = CROSS_IMAGE_VERIFY_PROMPT.format(text=text, img_num=img_idx + 1)
        # Feed: both full images + the crop
        response = self.vlm.generate_text([images[0], images[1], crop], prompt)
        return self._parse_verification(response)

    def _crop_image(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Crop image by bbox [x, y, w, h]. Returns None if invalid."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None

        # Clamp to image bounds
        img_w, img_h = image.size
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))

        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop((x1, y1, x2, y2))

    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """Parse verification JSON from VLM response."""
        if "assistant" in response:
            response = response.split("assistant")[-1]

        try:
            match = re.search(r'\{[^{}]*"valid"[^{}]*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "valid": bool(data.get("valid", False)),
                    "score": float(data.get("score", 0.0)),
                    "reason": str(data.get("reason", "")),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: check for yes/no keywords
        lower = response.lower()
        if "true" in lower or "yes" in lower:
            return {"valid": True, "score": 0.6, "reason": "parsed from text"}
        return {"valid": False, "score": 0.2, "reason": "parse failed"}
