# migrated to pyproject.toml
from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../pyproject.toml", globalns=globals())
author  # This name *looks* to be undefined, but it isn't.
