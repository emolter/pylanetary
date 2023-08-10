Description
***********

The idea behind pylanetary is to bring solar system science tools into the open-source Python 3 / Astropy ecosystem. We, and many of our colleagues, rely heavily on useful code snippets passed down from other solar system scientists. But these pieces of code are untested, in multiple languages, closed-source, and have many untracked dependencies. We want to fix that.

At present, two main packages are reasonably well-supported:

1. navigation: Tools to make and use ellipsoidal models of planets/large moons. This subpackage projects planet models into arbitrary observing geometries and pixel scales, compares those models with observational data, assigns latitudes, longitudes, and emission angles to observational data, and projects images onto latitude-longitude grids.
2. rings: Tools to model planetary ring systems.  This subpackage projects ring models into arbitrary observing geometries and pixel scales, compares those models with observational data, and makes radial and azimuthal profiles of observed rings.

The eventual goal is to become Astropy-affiliated, but that is a long way off. We would love your help developing it! See `Contributing to Pylanetary <https://pylanetary.readthedocs.io/en/latest/contributing.html>`_.