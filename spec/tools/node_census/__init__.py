"""Name the nodes of polars, the kinds of data and the input outcomes that the tests vary.

A node of the census is a call of polars with fixed arguments (`CensusCalls`),
built again with any input at each position (`CensusSurface`). Each input is
an outcome (`OutcomeSpace`): a column of one of the census kinds
(`CensusKinds`) in a row form, or a literal of a class. The samples of values
(`Sample`) give the data that polars runs on. The tests of the rules and the
check of random trees read these. Nothing here runs inside the plugin.
"""
