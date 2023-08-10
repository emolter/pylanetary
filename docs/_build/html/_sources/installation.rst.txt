Installation
************

* download the requirements.txt file using the "download raw" option
* ``conda create -n pylanetary-tester python=3.9`` (or any Python from 3.7 to 3.10)
* ``conda activate pylanetary-tester``
* ``pip install -r requirements.txt`` in same directory as requirements.txt
* ``pip install git+https://github.com/emolter/pylanetary.git@main``
	
Pylanetary relies on as-yet-unreleased versions of the image\_registration and astroquery packages, and pypi does not support installation of unreleased packages. This is the reason that simply pip install pylanetary will not work right.

Troubleshooting
~~~~~~~~~~~~~~~
Here we attempt to document install issues people have had, and their solutions

* requirements.txt not found. Most likely, you never downloaded requirements.txt, or you are not running pip install -r requirements.txt in the same folder where you put requirements.txt
* pip install requirements.txt fails instantly with a crazy-looking error message. Try opening requirements.txt, and check that its just ten or so lines of human-readable text. If it's some huge html soup, that means you didn't use the "download raw" button when downloading
* dependency conflicts when running pip install. Try using the option --force-reinstall, e.g., ``pip install --force-reinstall -r requirements.txt``
* dependency conflicts persist after force-reinstall. Check that your conda is working right. When this happened to a colleague, we found that ``which conda`` was returning some ten-line thing with a bunch of brackets instead of just the directory where conda lived. Solution would have been to completely dismantle and re-build conda, but that colleague decided it wasn't worth it (fair enough :)