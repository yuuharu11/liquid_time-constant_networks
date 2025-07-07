"""
Task modules for different ML tasks
"""

from .base import BaseTask
from .classification import MulticlassClassification

__all__ = [
    "BaseTask",
    "MulticlassClassification",
]
