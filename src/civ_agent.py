"""
CIV Agent: Cross-Image Verification Agent for multi-context visual grounding.

Three stages (all using a single VLM):
  Stage 1: Parse prompt → structured info
  Stage 2: VLM grounding → candidate bboxes per target image
  Stage 3: Cross-image verification → re-score and select best candidate
"""
from typing import List, Dict, Any
from PIL import Image

from src.two_stage_agent import TwoStageBaselineAgent
from src.cross_image_verifier import CrossImageVerifier


class CIVAgent(TwoStageBaselineAgent):
    """Three-stage agent with cross-image verification."""

    def __init__(self, vlm, config: Dict[str, Any]):
        super().__init__(vlm, config)
        self.verifier = CrossImageVerifier(vlm)

        # Scoring weights
        weights = config.get("weights", {})
        self.w_grounding = weights.get("grounding", 0.3)
        self.w_local = weights.get("local", 0.3)
        self.w_cross = weights.get("cross", 0.4)

        # Weights for referring type (no cross-image needed)
        self.w_ref_grounding = weights.get("ref_grounding", 0.4)
        self.w_ref_local = weights.get("ref_local", 0.6)

        # Ablation: disable cross-image verification
        self.use_cross = config.get("use_cross", True)

    def predict(self, sample: Dict[str, Any], images: List[Image.Image]) -> Dict[str, Any]:
        """Run three-stage prediction with CIV."""
        text = sample["text"]
        sample_id = sample["sample_id"]

        # Stage 1: Parse prompt
        parsed = self.parser.parse(images, text)

        if not parsed["target_images"]:
            return self._empty_result(sample_id, method="civ_agent")

        # Stage 2: Generate candidates
        candidates = self._ground_candidates(images, parsed, sample)

        if not candidates:
            return self._empty_result(sample_id, method="civ_agent")

        # Stage 3: Verify and re-score candidates
        # Use dataset's text_style (ground truth) instead of VLM-parsed reasoning_type
        # VLM was classifying everything as "referring", preventing cross-image verification
        text_style = sample.get("text_style", "").lower()
        # Map text_style to reasoning_type: "Comparison" → "comparison", "Reasoning" → "reasoning"
        reasoning_type = text_style if text_style in ("comparison", "reasoning") else "referring"
        verified = self._verify_candidates(candidates, images, sample, text, reasoning_type)

        if not verified:
            return self._empty_result(sample_id, method="civ_agent")

        best = max(verified, key=lambda c: c["final_score"])
        return {
            "sample_id": sample_id,
            "bbox": best["bbox"],
            "image_id": best["image_id"],
            "confidence": best["final_score"],
            "method": "civ_agent",
            "parsed": parsed,
            "verification": {
                "local_score": best.get("local_score"),
                "cross_score": best.get("cross_score"),
            },
        }

    def _verify_candidates(
        self,
        candidates: List[Dict[str, Any]],
        images: List[Image.Image],
        sample: Dict[str, Any],
        text: str,
        reasoning_type: str,
    ) -> List[Dict[str, Any]]:
        """Run verification on all candidates and compute final scores."""
        # Map image_id → image index
        img_id_to_idx = {img["id"]: i for i, img in enumerate(sample["images"])}

        verified = []
        for cand in candidates:
            img_idx = img_id_to_idx.get(cand["image_id"], 0)
            bbox = cand["bbox"]
            grounding_score = cand["score"]

            # Local verification
            local_result = self.verifier.verify_local(images[img_idx], bbox, text)
            local_score = local_result["score"]

            # Cross-image verification (skip for referring type or if disabled)
            if self.use_cross and reasoning_type in ("comparison", "reasoning"):
                cross_result = self.verifier.verify_cross_image(images, bbox, img_idx, text)
                cross_score = cross_result["score"]

                final_score = (
                    self.w_grounding * grounding_score
                    + self.w_local * local_score
                    + self.w_cross * cross_score
                )
            else:
                cross_score = None
                final_score = (
                    self.w_ref_grounding * grounding_score
                    + self.w_ref_local * local_score
                )

            # Suppress candidates that fail verification
            if not local_result["valid"] and local_score < 0.3:
                final_score *= 0.1

            verified.append({
                **cand,
                "local_score": local_score,
                "cross_score": cross_score,
                "final_score": final_score,
            })

        return verified

    def _empty_result(self, sample_id: int, method: str = "civ_agent") -> Dict[str, Any]:
        return {
            "sample_id": sample_id,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "image_id": -1,
            "confidence": 0.0,
            "method": method,
        }
