
Benchmarking Denoisers for Plug-and-Play Priors
===============================================

|Build Status| |Python 3.10+|

The goal of this benchmark is to compare the performance of different denoisers
in the context of plug-and-play priors. Denoisers are methods
that take an image and a noise level as input and output a denoised image.
They are usually trained by minimizing the following objective function:

$$
\\min_{\\theta} \\sum_{i=1}^n \\|D_\\theta(X_i + \\sigma_i w_i, \sigma_i) - X_i\\|^2
$$

where the $X_i$ are clean images, $w_i$ is a noise noise, $\sigma_i$ is the noise level, and $D_\\theta$ is the denoiser parameterized by $\theta$.
The idea of this benchmark is to compare the denoising performances of
different denoisers, and relate this performances with the performance of
plug-and-play algorithms using these denoisers.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/benchmark_denoisers
   $ benchopt run benchmark_denoisers

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_denoisers -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/benchmark_denoisers/actions/workflows/main.yml/badge.svg
   :target: https://github.com/tomMoral/benchmark_denoisers/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
