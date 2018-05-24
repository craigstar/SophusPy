# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import eigency

setup(
    ext_modules = cythonize([Extension("sophus", 
        sources=["sophus.pyx"],
        include_dirs = [".", "./sophus"] + eigency.get_includes(include_eigen=True),
        # include_dirs = [".", "/Users/craigstar/Applications/anaconda2/include/eigen3", "./Sophus"],
        language="c++",
        extra_compile_args=["-std=c++11", "-stdlib=libc++"])])
)
