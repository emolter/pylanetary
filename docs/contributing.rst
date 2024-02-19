Contributing
************

Thanks for considering making a contribution!

Contribution Workflow
---------------------
If you are considering making a contribution, please follow the steps below. The tutorial on the `GitHub flow <https://docs.github.com/en/get-started/quickstart/github-flow>`_ may be helpful if you are new to git and GitHub.

 * Open an issue `here <https://github.com/emolter/pylanetary/issues>`_ describing the problem you intend to solve or the new feature you intend to build. We are pretty lax about how issues are presented or formatted, but it still might be useful to look through `the issues quickstart <https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart>`_. 
 
 * Fork the main branch of the repository. See the `contributing quickstart <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_ for the recommended workflow and git commands. See the `forking quickstart <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ for a more detailed tutorial about forking, including setup for synchronizing with the upstream branch.
 
 * Make your changes or new development. Please see the New Features section below, and start a conversation in the comments of the issue you submitted if you have questions about how best to implement your changes.
 
 * Open a draft pull request. We strongly encourage you to do this early in development. The workflow to do this is also in the `contributing quickstart <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_. It is helpful if you `link the pull request to the issue you opened <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`_. 
 
 * When you are ready, request a review of the pull request. Please understand that changes are nearly always requested upon first review.

New Features
------------
We welcome new features!

Tests are highly encouraged, but can be built up over time.  At least one
example usage is necessary for a new module to be accepted.

The minimum requirements for a new feature are:

 * Open a discussion on the Issues page and receive feedback about how your new feature can best be implemented.

 * If your feature is relatively large and doesn't fit anywhere else, make a new subdirectory of pylanetary with at least an
   ``__init__.py`` and a ``core.py``::
 
     pylanetary/feature
     pylanetary/feature/__init__.py
     pylanetary/feature/core.py

 * Add a ``tests/`` directory with at least one test::
 
     pylanetary/feature/tests
     pylanetary/feature/tests/__init__.py
     pylanetary/feature/tests/test_feature.py

 * Add documentation within the docstring with at least one example. Try to conform to the `numpy docstyle <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__.
	 
 * For any major new functionality or workflow, make an example Jupyter notebook::
 
     docs/tutorials/feature-tutorial.ipynb

Important Guidelines
--------------------
Pylanetary intends to provide *generic* tools for solar system data processing and modeling.
As such, all functions and classes should remain agnostic to planet, observatory, 
wavelength band, etc. Defaults for a given planet, observatory, wavelength band, etc.
may be provided as .yaml (preferred) or another text file format, and should go in the ``pylanetary/feature/data`` subdirectory. See ``pylanetary/rings/data`` for an example.

Features that are primarily relevant to a single observatory, planet, wavelength band, etc.
should be placed in separate GitHub repositories and import pylanetary. If portions 
of that functionality are generic, you could consider including them in pylanetary and then writing a separate, more specific package that imports pylanetary.

Docstrings should be written for every function and should adhere to the `numpy docstyle <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_
format.

Dependencies
------------
The astropy ecosystem tools should be used whenever possible.
For example, `astropy.table` should be used for table handling,
or `astropy.units` for unit and quantity
handling. This is aspirational at time of writing, but good to keep in mind.

If a new contribution brings along any additional dependencies, the necessity
of those dependencies must be well-justified, the dependencies should provide 
functionality that does not exist in the astropy ecosystem, and 
those dependencies must be approved by an approver on the Pylanetary team.
This is another good reason to make a draft pull request early on in the 
development process.

Imposter syndrome disclaimer
----------------------------
We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
pylanetary based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.