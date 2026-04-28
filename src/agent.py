"""
Two-Stage Agent with Cross-Image Verification for Visual Grounding
"""
import json
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image

from src.vlm_inference import Qwen2VLLM, load_vlm


class AgentConfig(TypedDict):
    """Agent configuration."""
    model_name: str
    device: str
    cache_dir: str | None
    use_verification: bool
    verification_iterations: int
    confidence_threshold: float


class AgentPrediction(TypedDict):
    """Agent prediction result."""
    sample_id: int
    bbox: list[float]
    image_id: int
    method: str
    confidence: float
    verification_iterations: int
    raw_response: str | None


class CrossImageVerificationAgent:
    """
    Two-stage agent for visual grounding with cross-image verification.
    
    Stage 1: Initial localization using both images
    Stage 2: Cross-image verification to confirm/deny the prediction
    """
    
    SYSTEM_PROMPT = """You are an expert visual grounding assistant. Your task is to:
1. Analyze both images carefully
2. Determine which image (first or second) contains the target described in the text
3. Locate the target object and output its bounding box

Text description: {text}

Output format (JSON):
{{
  "target_image": 1 or 2,
  "bbox": [x1, y1, x2, y2] in pixel coordinates (top-left and bottom-right corners),
  "confidence": 0.0-1.0
}}

Guidelines:
- The text may explicitly mention "first image" or "second image"
- If no target found in either image, output: {{"target_image": 0, "bbox": [0, 0, 0, 0], "confidence": 0.0}}
- Bounding box format: [x1, y1, x2, y2] where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner"""
    
    VERIFICATION_PROMPT = """Given the text description and your initial prediction, verify if the bounding box is correct.

Text: {text}
Initial prediction: {bbox}

Consider:
1. Is the bbox location accurate for the described target?
2. Which image contains the target (first or second)?
3. Is the bbox size appropriate?

Output only the refined bbox [x, y, w, h] or [0, 0, 0, 0] if incorrect."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.vlm = None
    
    def load(self):
        """Load VLM model."""
        self.vlm = load_vlm(self.config)
    
    def predict(
        self,
        sample: dict[str, Any],
        images: list[Image.Image],
    ) -> AgentPrediction:
        """
        Make prediction for a sample.
        
        Args:
            sample: Sample dict from dataset
            images: List of 2 PIL Images
        
        Returns:
            AgentPrediction with bbox and metadata
        """
        if self.vlm is None:
            self.load()
        
        text = sample["text"]
        sample_id = sample["sample_id"]
        
        if self.config.get("use_verification", True):
            return self._predict_with_verification(sample, images, text, sample_id)
        else:
            return self._predict_direct(sample, images, text, sample_id)
    
    def _predict_direct(
        self,
        sample: dict[str, Any],
        images: list[Image.Image],
        text: str,
        sample_id: int,
    ) -> AgentPrediction:
        """Direct prediction without verification."""
        result = self.vlm.predict_bbox(
            images=images,
            text=text,
            prompt=self.SYSTEM_PROMPT.format(text=text),
        )

        bbox = result["bbox"]
        target_image = result.get("target_image", 1)

        # Map target_image (1 or 2) to actual image_id
        # Default to first image if invalid
        if target_image == 2:
            image_id = sample["images"][1]["id"]
        else:
            image_id = sample["images"][0]["id"]

        return {
            "sample_id": sample_id,
            "bbox": bbox,
            "image_id": image_id,
            "method": "direct",
            "confidence": result.get("confidence", 1.0),
            "verification_iterations": 0,
            "raw_response": result.get("raw_response"),
        }
    
    def _predict_with_verification(
        self,
        sample: dict[str, Any],
        images: list[Image.Image],
        text: str,
        sample_id: int,
    ) -> AgentPrediction:
        """Prediction with cross-image verification loop."""
        max_iterations = self.config.get("verification_iterations", 2)

        current_bbox = None
        target_image = 1
        raw_response = None

        for iteration in range(max_iterations):
            if iteration == 0:
                result = self.vlm.predict_bbox(
                    images=images,
                    text=text,
                    prompt=self.SYSTEM_PROMPT.format(text=text),
                )
                current_bbox = result["bbox"]
                target_image = result.get("target_image", 1)
                raw_response = result.get("raw_response")
            else:
                verification_prompt = self.VERIFICATION_PROMPT.format(
                    text=text,
                    bbox=current_bbox,
                )
                result = self.vlm.predict_bbox(
                    images=images,
                    text=f"{text}\n\nPrevious prediction: {current_bbox}\n\nRefine if needed:",
                    prompt="Verify and refine the bbox if needed:",
                )
                new_bbox = result["bbox"]

                if new_bbox == [0, 0, 0, 0]:
                    current_bbox = [0, 0, 0, 0]
                elif new_bbox != current_bbox:
                    current_bbox = new_bbox
                    target_image = result.get("target_image", target_image)
                raw_response = result.get("raw_response")

            if current_bbox == [0, 0, 0, 0]:
                break

            if self._check_confidence(result.get("confidence", 1.0)):
                break

        # Map target_image to actual image_id
        # Default to first image if invalid
        if target_image == 2:
            image_id = sample["images"][1]["id"]
        else:
            image_id = sample["images"][0]["id"]

        return {
            "sample_id": sample_id,
            "bbox": current_bbox,
            "image_id": image_id,
            "method": "verification",
            "confidence": 1.0,
            "verification_iterations": iteration + 1,
            "raw_response": raw_response,
        }
    
    def _get_target_image_id(
        self,
        sample: dict[str, Any],
        bbox: list[float],
    ) -> int:
        """Determine which image contains the target."""
        if bbox == [0, 0, 0, 0]:
            return -1
        
        images = sample.get("images", [])
        
        if len(images) == 0:
            return -1
        
        # If we have target positions from the dataset, use them
        # Target positions indicate which image(s) contain the target (0 for first image, 1 for second)
        target_positions = sample.get("target_positions", [])
        if target_positions:
            # If target is in specific image(s), return the first one
            # In case of multiple target positions, we pick the first one
            target_image_index = target_positions[0]
            if 0 <= target_image_index < len(images):
                return images[target_image_index]["id"]
        
        # Fallback: return first image (should not happen with proper dataset)
        return images[0]["id"]
    
    def _check_confidence(self, confidence: float) -> bool:
        """Check if confidence is above threshold."""
        threshold = self.config.get("confidence_threshold", 0.8)
        return confidence >= threshold


class BatchAgent:
    """Batch processing wrapper for the agent."""
    
    def __init__(self, agent: CrossImageVerificationAgent):
        self.agent = agent
    
    def predict_batch(
        self,
        samples: list[dict[str, Any]],
        dataset,
        output_path: str = None,
        verbose: bool = True,
    ) -> list[AgentPrediction]:
        """
        Run prediction on a batch of samples.
        
        Args:
            samples: List of sample dicts
            dataset: MCBenchDataset instance for loading images
            output_path: Optional path to save results
            verbose: Print progress
        
        Returns:
            List of AgentPrediction results
        """
        results = []
        total = len(samples)
        
        for idx, sample in enumerate(samples):
            if verbose and (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{total}")
            
            images = dataset.load_images(sample)
            
            try:
                prediction = self.agent.predict(sample, images)
                results.append(prediction)
            except Exception as e:
                print(f"Error on sample {sample.get('sample_id', idx)}: {e}")
                results.append({
                    "sample_id": sample.get("sample_id", idx),
                    "bbox": [0, 0, 0, 0],
                    "image_id": -1,
                    "method": "error",
                    "confidence": 0.0,
                    "verification_iterations": 0,
                    "raw_response": str(e),
                })
            
            if output_path and (idx + 1) % 50 == 0:
                self._save_intermediate(results, output_path)
        
        if output_path:
            self._save_intermediate(results, output_path)
        
        return results
    
    def _save_intermediate(self, results: list, output_path: str):
        """Save intermediate results."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def create_agent(config: dict) -> CrossImageVerificationAgent:
    """Create agent from config."""
    agent_config: AgentConfig = {
        "model_name": config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "device": config.get("device", "cuda"),
        "cache_dir": config.get("cache_dir", None),
        "use_verification": config.get("use_verification", True),
        "verification_iterations": config.get("verification_iterations", 2),
        "confidence_threshold": config.get("confidence_threshold", 0.8),
    }
    return CrossImageVerificationAgent(agent_config)
