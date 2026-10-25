"""Tests of the engine, the part of the plugin that decides answers with no mypy involved.

The Python models in `spec/tests` are the executable form of the
design. They are small and slow. The engine is the fast code that the
plugin runs. Several modules here build the same values on both sides
and check that both give the same answer to every question in a small
fixed space. Without these checks, the engine can drift away from the
design and no test notices.

Other modules check the codecs, which turn engine values into mypy
types and back, with no full mypy build.
"""
