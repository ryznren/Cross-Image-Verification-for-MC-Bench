"""
VLM Inference Wrapper for Qwen2.5-VL
"""
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen2VLLM:
    """Qwen2.5-VL inference wrapper."""
    
    DEFAULT_PROMPT = """You are an expert visual grounding assistant. Given two images and a text description, 
your task is to locate the target object described in the text by outputting a bounding box.
The bounding box should be in format: [x, y, width, height] where values are normalized (0-1).

Output format: 
- If target found: [x, y, w, h]
- If no target: [0, 0, 0, 0]

Text: {text}

Provide only the bounding box coordinates, no other text."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        
        self.processor = None
        self.model = None
        self._loaded = False
    
    def load(self):
        """Load model and processor."""
        if self._loaded:
            return
        
        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model.eval()
        self._loaded = True
        print(f"Model loaded on {self.device}")
    
    def predict_bbox(
        self,
        images: list[Image.Image],
        text: str,
        prompt: str = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Predict bounding box for given images and text.

        Args:
            images: List of 2 PIL Images
            text: Text description
            prompt: Optional custom prompt
            temperature: Sampling temperature (0 for greedy)

        Returns:
            Dict with 'bbox', 'target_image', 'raw_response', 'confidence'
        """
        if not self._loaded:
            self.load()

        if prompt is None:
            prompt = self.DEFAULT_PROMPT.format(text=text)

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        bbox, target_image = self._parse_bbox_with_image(response)

        return {
            "bbox": bbox,
            "target_image": target_image,
            "raw_response": response,
            "confidence": 1.0 if temperature == 0 else None,
        }

    def generate_text(
        self,
        images: list[Image.Image],
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text response for given images and prompt.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if not self._loaded:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return response

    def _parse_bbox(self, response: str) -> list[float]:
        """Parse bounding box from model response.

        Model outputs [x1, y1, x2, y2] in pixel coordinates.
        Convert to COCO [x, y, width, height] format (keep pixel coordinates).
        """
        import re

        response = response.strip()

        patterns = [
            r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',
            r'"bbox_2d":\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',
            r'([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                nums = [float(x) for x in matches[-1]]
                if len(nums) == 4:
                    x1, y1, x2, y2 = nums
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        return [x1, y1, w, h]

        return [0.0, 0.0, 0.0, 0.0]

    def _parse_bbox_with_image(self, response: str) -> tuple[list[float], int]:
        """Parse bbox and target image from JSON response.

        Returns:
            (bbox, target_image) where target_image is 1 or 2
        """
        import re
        import json as json_lib

        # Extract only the assistant's response (after last "assistant" marker)
        if "assistant" in response:
            response = response.split("assistant")[-1]

        # Try to parse JSON format
        try:
            json_match = re.search(r'\{[^}]*"target_image"[^}]*\}', response, re.DOTALL)
            if json_match:
                data = json_lib.loads(json_match.group())
                target_image = data.get("target_image", 1)
                bbox = data.get("bbox", [0, 0, 0, 0])

                # Ensure bbox is list of floats
                bbox = [float(x) for x in bbox]

                # If target_image is 0, treat as no target found
                if target_image == 0:
                    return [0.0, 0.0, 0.0, 0.0], 1

                # Convert [x1,y1,x2,y2] to [x,y,w,h]
                # Model is instructed to output [x1,y1,x2,y2] format
                if len(bbox) == 4 and bbox != [0, 0, 0, 0]:
                    x1, y1, x2, y2 = bbox
                    # Only convert if x2 > x1 and y2 > y1 (valid coordinates)
                    if x2 > x1 and y2 > y1:
                        bbox = [x1, y1, x2 - x1, y2 - y1]

                return bbox, target_image
        except:
            pass

        # Fallback: use old parsing
        bbox = self._parse_bbox(response)
        return bbox, 1  # Default to first image
    
    def predict_with_verification(
        self,
        images: list[Image.Image],
        text: str,
        verification_prompt: str = None,
    ) -> dict[str, Any]:
        """
        Two-stage prediction with cross-image verification.
        
        Stage 1: Initial prediction
        Stage 2: Verify and refine the prediction
        """
        if not self._loaded:
            self.load()
        
        initial_result = self.predict_bbox(images, text)
        
        if verification_prompt is None:
            verification_prompt = f"""Given the text: "{text}"

The predicted bounding box is: {initial_result['bbox']}

Please verify if this bounding box is correct for the described target.
Consider both images and determine which image contains the target.

Output format:
- If correct: [x, y, w, h] (the same or refined box)
- If incorrect: [0, 0, 0, 0] (no target found)

Provide only the bounding box coordinates."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": verification_prompt},
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
            )
        
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        refined_bbox = self._parse_bbox(response)

        return {
            "initial_bbox": initial_result["bbox"],
            "refined_bbox": refined_bbox,
            "raw_response": response,
        }


def load_vlm(config: dict) -> Qwen2VLLM:
    """Load VLM from config."""
    vlm = Qwen2VLLM(
        model_name=config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        device=config.get("device", "cuda"),
        cache_dir=config.get("cache_dir", None),
    )
    return vlm
