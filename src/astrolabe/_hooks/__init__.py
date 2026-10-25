"""The mypy hooks of the plugin, with one module for each family of operations.

A user never imports this package at run time. The public names live in
the top-level `astrolabe` package. The code here runs only inside mypy,
when the checker reads a call to a frame method, an expression method or
a polars function. The entry point in `plugin.py` only routes each call
to its hook, so it holds none of the rules that decide a result.
"""
