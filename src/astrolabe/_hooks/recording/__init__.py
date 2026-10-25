"""The recorder: the hooks that write each call on an expression as code, in a deferred type.

A deferred type holds the code of an expression, as the user wrote it with the
public API of polars (`astrolabe._hooks.deferred`). A few generic hooks write
that code: one for a method or an operator of an expression, one for a member
of a namespace such as `str`, one for a function of `pl` and one for a selector.
Each hook writes what the user wrote, and knows nothing about any node of
polars. The checker computes the data types from the code only when the
expression meets a frame or a declaration.
"""
