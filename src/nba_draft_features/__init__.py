"""
nba_draft_features
===================

Reusable data-cleaning and feature-engineering utilities for the 36120 AT1
NBA sustained-career-prediction assignment. Extracted from experiment
notebooks so the same, tested logic can be shared across every experiment
notebook and reused by teammates.
"""

from .cleaning import ht_to_inches, clean_height, clean_class_year
from .features import (
    add_age_at_season,
    add_per40_rates,
    add_shot_profile_ratios,
    add_interaction_features,
)

__version__ = "0.2.0"

__all__ = [
    "ht_to_inches",
    "clean_height",
    "clean_class_year",
    "add_age_at_season",
    "add_per40_rates",
    "add_shot_profile_ratios",
    "add_interaction_features",
]
