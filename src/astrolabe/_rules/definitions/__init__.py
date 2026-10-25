"""The definitions of the nodes of polars: for each node, the parts of the vocabulary that it uses.

Polars declares the rules of each node in a small vocabulary, and the other
modules of `astrolabe._rules` translate that vocabulary. A node has rules at
four places: its type function, its options, its cast pass and its kernel. A
definition names, at each place, the part of the vocabulary that polars names
there, for example `FieldsMapper.with_same_dtype` as the type function of
`abs`. A definition computes nothing of its own. Code that evaluates a node
reads the definition and calls the parts that it names.

The package has these modules:

- `node_options`: the identity of a node, and the options of a node, read
  from the tree that polars serializes.
- `type_function_adapters`, `cast_adapters` and `kernel_adapters`: one small
  adapter for each shape that a part has at the type function, at the cast
  pass and at the kernel.
- `definition`: the definition of one node.
- `elementwise`, `binary_operators`, `aggregations` and `length_preserving`:
  the definitions of one family of nodes each.
- `strings`, `temporal` and `structs`: the definitions of the functions of
  one namespace each.
- `overrides`: the rules written for one node, where polars does what no
  part of the vocabulary states, each with its reason and its probe.
- `registry`: every definition, by the identity of its node.
"""
