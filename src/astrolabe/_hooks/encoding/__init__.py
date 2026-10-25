"""Turn engine values into mypy types, and read those types back.

The engine decides every rule of the checker, and it works on plain Python
values that know nothing about mypy. mypy knows only types. This package
is the one crossing between the two. Each codec here writes one kind of
engine value as a mypy type, and reads that type back into the same value.

The value must travel inside the type. mypy keeps a cache of the types of
each module that did not change, and on a later run it loads those types
without running the plugin again. A value kept anywhere beside the type
is present on the first run and missing on the second.

The type is built from marker classes: small generic classes in the public
modules whose type arguments hold the parts of the value. What mypy shows
for such a type is the encoded value itself, so an error message shows
what the user wrote. A codec writes only the normal form of a value, and
no other code builds these types by hand. As a result, two equal values
always give the same type, and mypy can compare them without help.
"""
