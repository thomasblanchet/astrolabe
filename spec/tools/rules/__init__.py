"""Compare the rules of `astrolabe._rules` with polars over random trees of the census nodes.

`random_trees.py` builds the trees and runs polars on them, `tree_check.py`
runs them through the evaluator of whole trees and counts the differences,
and `census_text.py` gives the leaf of the rules for a census outcome and an
answer of the rules as text.
"""
