# migrated to pyproject.toml
from sphinx_pyproject import SphinxConfig
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

config = SphinxConfig("../pyproject.toml", globalns=globals())
