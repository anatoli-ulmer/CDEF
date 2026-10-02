#!/usr/bin/env python3

import os
import tempfile

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


def _check_compile_flag(compiler, flag, script):
    """Return True if *compiler* accepts *flag* for the supplied C source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = os.path.join(tmpdir, "flagtest.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(script)

        try:
            compiler.compile([source], output_dir=tmpdir, extra_postargs=[flag])
        except Exception:
            return False

    return True


def check_for_openmp(compiler):
    """Return compile and link flags for OpenMP when supported."""
    test_program = r"""
        #ifdef _OPENMP
        #include <omp.h>
        #else
        #error No OpenMP support
        #endif
        int main(void) {
            int n = 0;
            #pragma omp parallel reduction(+:n)
            n += 1;
            return n < 1;
        }
    """

    if compiler.compiler_type == "msvc":
        for flag in ("/openmp", "/openmp:experimental"):
            if _check_compile_flag(compiler, flag, test_program):
                return [flag], []
        return [], []

    if _check_compile_flag(compiler, "-fopenmp", test_program):
        return ["-fopenmp"], ["-fopenmp"]

    return [], []


class build_ext(_build_ext):
    def build_extensions(self):
        print("Checking for OpenMP support...\t", end="")
        compile_flags, link_flags = check_for_openmp(self.compiler)
        print(" ".join(compile_flags) if compile_flags else "not available")

        if not compile_flags:
            print(
                "WARNING: OpenMP support is not available in the default C compiler; "
                "Debyer will run on a single core."
            )

        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_flags
            ext.extra_link_args = list(ext.extra_link_args or []) + link_flags

        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "CDEF.debyer",
            [
                "src/debyer/atomtables.c",
                "src/debyer/debyer.c",
                "src/debyer/debyer_wrap.c",
                "src/debyer/polyhedrongeom.c",
            ],
            include_dirs=[numpy.get_include()],
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
