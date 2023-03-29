data processing and modeling tools for ring, moon, and planet observations
--------------------------------------------------------------------------

.. image:: https://zenodo.org/badge/459414964.svg
   :target: https://zenodo.org/badge/latestdoi/459414964

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
	
.. image:: https://zenodo.org/badge/459414964.svg
   :target: https://zenodo.org/badge/latestdoi/459414964

Quickstart
----------
We plan to make this pip installable with the first (pre-alpha) release in the next few weeks. For now, check out the capabilities we're working on in the Jupyter notebooks under "notebooks"

Scope and Goal
--------------
The idea behind pylanetary is to bring solar system science tools into the open-source Python 3 / Astropy ecosystem. We, and many of our colleagues, rely heavily on useful code snippets passed down from other solar system scientists. But these pieces of code are untested, in multiple languages (IDL, Python 2, etc.), closed-source, and have many untracked dependencies. We want to fix that.

At present, we are working on two main packages:
1. planetnav: Tools to make and use ellipsoidal models of planets/large moons. This subpackage projects planet models into arbitrary observing geometries and pixel scales, compares those models with observational data, assigns latitudes, longitudes, and emission angles to observational data, and projects images onto latitude-longitude grids.
2. rings: Tools to model planetary ring systems.  This subpackage projects ring models into arbitrary observing geometries and pixel scales, compares those models with observational data, and makes radial and azimuthal profiles of observed rings.

The eventual goal is to become Astropy-affiliated, but that is a long way off.
At present, this repository is just a skeleton. We would love your help developing it!  See Contributing.

License
-------

This project is Copyright (c) Ned Molter & Chris Moeckel and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Contributing
------------

We love contributions! pylanetary is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

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
