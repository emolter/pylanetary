Contributing to Pylanetary
==========================
Please see `astropy's contributing guildelines
<http://www.astropy.org/contribute.html>`__ for a general guide to the
workflow involving git, etc.  Everything below is astroquery-specific.

We strongly encourage draft pull requests to be opened early in development.
If you are thinking of contributing a new module, please open an issue marked "enhancement"
describing your feature idea, and please open a pull request
as soon as you start developing code and mark it as a Draft PR on github.

New Features
------------
We welcome any and all new features!

Tests are highly encouraged, but can be built up over time.  At least one
example use is necessary for a new module to be accepted.

The minimum requirements for a new feature are:

 * Add the feature as a subdirectory of pylanetary with at least an
   ``__init__.py`` and a ``core.py``::
 
     pylanetary/feature
     pylanetary/feature/__init__.py
     pylanetary/feature/core.py

 * Add a ``tests/`` directory with at least one test::
 
     pylanetary/feature/tests
     pylanetary/feature/tests/__init__.py
     pylanetary/feature/tests/test_feature.py

 * Add some documentation - at least one example, but it can be sparse at first::
 
     docs/pylanetary/feature.rst
	 
 * For any major new functionality or workflow, make an example Jupyter notebook::
 
     notebooks/feature-tutorial.ipynb


Important Guidelines
--------------------
 Pylanetary intends to provide *generic* tools for solar system data processing and modeling.
 As such, all functions and classes should remain agnostic to planet, observatory, 
 wavelength band, etc. Defaults for a given planet, observatory, wavelength band, etc.
 may be provided as .yaml (preferred) or another text file format, and should go in the ``pylanetary/feature/data`` subdirectory. See ``pylanetary/rings/data`` for an example.
 
 Features that are primarily relevant to a single observatory, planet, wavelength band, etc.
 should be placed in separate GitHub repositories and import pylanetary. Again, if portions 
 of that functionality are generic, they can be included in pylanetary 
 
 Docstrings should be written for every function and should adhere to the same
 format as shown in the functions in ``pylanetary/feature/planetnav/core.py``


Dependencies
------------
The astropy ecosystem tools should be used whenever possible.
For example, `astropy.table` should be used for table handling,
or `astropy.units` for unit and quantity
handling.

If a new contribution brings along any additional dependencies, the necessity
of those dependencies must be well-justified, the dependencies should provide 
functionality that does not exist in the astropy ecosystem, and 
those dependencies must be approved by an approver on the Pylanetary team.
This is another good reason to make a draft pull request early on in the 
development process.