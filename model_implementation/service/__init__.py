"""
Service package — shared utilities for all classification models.

Exports:
    load_data           : Load the raw CSV dataset.
    preprocess_data     : Preprocess a raw DataFrame.
    get_preprocessed_data : One-shot load + preprocess helper.
"""

from model_implementation.service.data_service import (  # noqa: F401
    load_data,
    preprocess_data,
    get_preprocessed_data,
)
