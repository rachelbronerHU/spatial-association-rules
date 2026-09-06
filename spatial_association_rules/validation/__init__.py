"""
Deciding whether a rule is real.

significance.py     does a rule survive when the labels are shuffled?
false_discovery.py  many rules were tested at once: correct for that
"""

from .significance import p_values_for, seed_for

__all__ = ["p_values_for", "seed_for"]
