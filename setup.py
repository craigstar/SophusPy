import os
import sys
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

__author__ = 'Craigstar'
__date__ = '2019/06/15'


class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        # example of cmake args
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DPYTHON_EXECUTABLE=' + sys.executable # make sure cmake uses the same python version
        ]

        # example of build args
        build_args = [
            '--config', 'Release',
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sophuspy",
    version="0.0.8",
    author="Craigstar",
    author_email="work.craigzhang@gmail.com",
    keywords="Lie Group",
    license="MIT",
    description=("A python binding using pybind11 for Sophus which is a C++ Lie library.(SO3 && SE3)"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/craigstar/SophusPy",
    packages=['sophus'],
    install_requires=['numpy', 'pybind11'],
    include_package_data=True,
    platforms=["all"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=[CMakeExtension('sophus/sophuspy')],
    cmdclass={
        'build_ext': build_ext,
    }
)