from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__author__ = 'Craigstar'
__date__ = '2024/02/21'
__version__ = "0.1.0"


ext_modules = [
    Pybind11Extension(
        "sophuspy",
        ["sophuspy/sophuspy.cpp"],
        include_dirs=[
            "eigen3",
            "sophuspy",
            "sophuspy/include/original",
            "sophuspy/include/extension",
        ],
        language='c++',
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sophuspy",
    version=__version__,
    author="Craigstar",
    author_email="work.craigzhang@gmail.com",
    keywords="Lie Group",
    license="MIT",
    description=("A python binding using pybind11 for Sophus which is a C++ Lie library.(SO3 && SE3)"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/craigstar/SophusPy",
    packages=['sophuspy'],
    install_requires=['numpy'],
    include_package_data=True,
    platforms=["all"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': build_ext,
    }
)