"""
MC-Bench: Cross-Image Verification for Multi-Context Visual Grounding
"""
from src.dataset import MCBenchDataset
from src.agent import CrossImageVerificationAgent, create_agent, BatchAgent

__all__ = [
    "MCBenchDataset",
    "CrossImageVerificationAgent",
    "create_agent",
    "BatchAgent",
]
