"""Build the embedded MIC and strict-RD extensions on Linux/GCC."""
import sys
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

if not sys.platform.startswith('linux'):
    raise RuntimeError('CellFate is validated on Linux with GCC/OpenMP.')

native = 'src/cellfate/_native'
mic = Extension(
    'cellfate._native.mine',
    [f'{native}/mine.pyx', f'{native}/libmine/mine.c'],
    include_dirs=[numpy.get_include(), native],
    libraries=['m'],
    extra_compile_args=['-Wall', '-O2', '-fno-fast-math', '-ffp-contract=off', '-fopenmp'],
    extra_link_args=['-fopenmp'],
)
rd = Extension(
    'cellfate._native.register_product',
    [f'{native}/register_product.c'],
    extra_compile_args=['-O3', '-march=native', '-fno-fast-math', '-ffp-contract=off'],
)
setup(ext_modules=cythonize([mic], force=True,
    compiler_directives={'language_level': 3}) + [rd])
