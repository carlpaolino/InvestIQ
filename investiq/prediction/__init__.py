"""
Prediction engine module for InvestIQ.

Handles real-time predictions, suggestion generation, and API endpoints.
"""

from .predictor import predict_return, generate_suggestion
from .cli import main as cli_main
from .api import create_app

__all__ = [
    "predict_return",
    "generate_suggestion", 
    "cli_main",
    "create_app"
]
