========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/python-yggdrasil/badge/?style=flat
    :target: https://readthedocs.org/projects/python-yggdrasil
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/jxnl/python-yggdrasil.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/jxnl/python-yggdrasil

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/jxnl/python-yggdrasil?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/jxnl/python-yggdrasil

.. |requires| image:: https://requires.io/github/jxnl/python-yggdrasil/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/jxnl/python-yggdrasil/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/jxnl/python-yggdrasil/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/jxnl/python-yggdrasil

.. |version| image:: https://img.shields.io/pypi/v/yggdrasil.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/yggdrasil

.. |commits-since| image:: https://img.shields.io/github/commits-since/jxnl/python-yggdrasil/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/jxnl/python-yggdrasil/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/yggdrasil.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/yggdrasil

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/yggdrasil.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/yggdrasil

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/yggdrasil.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/yggdrasil


.. end-badges

Finetuning for multitask siamese networks

* Free software: MIT license

Installation
============

::

    pip install yggdrasil

Documentation
=============

https://python-yggdrasil.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
