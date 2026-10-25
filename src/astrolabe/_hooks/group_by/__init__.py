"""Type a group-by: `df.group_by(...)`, then `.agg(...)` or a short form such as `.sum()`.

The `group_by` call gives a typed group-by that holds the key columns and the columns of the
frame. The call after it reads them back and builds the typed frame that it gives, with the keys
first. `group_by.py` types the first call, `agg.py` types `.agg`, and `shorthand.py` types the
short forms.
"""
