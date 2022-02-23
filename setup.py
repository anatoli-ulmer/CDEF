#! /usr/bin/env python3
import os
import sys
import sysconfig
import platform
import subprocess
import tempfile
import shutil

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

import distutils.ccompiler
import distutils.errors


def ac_check_flag(flags, script):
    # emulate AC_CHECK_FLAG from autotools to test if the compiler
    # supports a given flag. Return the first working flag. 

    # Get compiler invocation
    
    compiler = distutils.ccompiler.new_compiler()
    distutils.sysconfig.customize_compiler(compiler)
    
    for flag in flags:
        # Create a temporary directory
        tmpdir = tempfile.mkdtemp()
        curdir = os.getcwd()
        os.chdir(tmpdir)

        # Attempt to compile the test script.
        filename = r'flagtest.c'
        with open(filename,'w') as f :
            f.write(script)
        
        try:
            compiler_result = compiler.compile(['flagtest.c'], extra_postargs=[flag])
            success = True
        except distutils.errors.CompileError:
            success = False

        # Clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
        
        if success:
            return flag

    return ""


def check_for_openmp_simd():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
        
    omptestprog = '''
        #ifdef _OPENMP
        #include <omp.h>
        #else
        #error No OpenMP support
        #endif
        #include <stdio.h>
        int main() {
            #pragma omp parallel
            printf("Hello from thread %d, nthreads %d\\n", omp_get_thread_num(), omp_get_num_threads());
            float a[4] = { 0, 1, 2, 3};
            int i=0; float sum = 0.0f;
            #pragma omp simd reduction (+:sum)
            for (i=0; i<4; i++)
                sum += a[i];
        }
'''
    
    ompflags = ['-fopenmp', '/openmp', '/openmp:experimental']
    
    ompflag  = ac_check_flag(ompflags, omptestprog);

    if ompflag == "":
        return []

    return [ompflag]



class build_ext(_build_ext):
    # find openMP options, if available
    def finalize_options(self):
        _build_ext.finalize_options(self)
        print("Checking for OpenMP support...\t", end="")

        extraompflag = check_for_openmp_simd()

        print(" ".join(extraompflag))

        if extraompflag == []:
            print ("""WARNING
        OpenMP support is not available in your default C compiler
        The program will only run on a single core. 
        """)
        
        for ext in self.extensions:
            ext.extra_compile_args.extend(extraompflag)
            ext.extra_link_args.extend(extraompflag)
            print("Current value: ",ext.extra_compile_args)
            pass


numpyinclude = numpy.get_include()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='CDEF',
    version='1.0',
    author='Jerome Deumer',
    author_email='jerome.deumer@ptb.de',
    description='Compute small-angle scattering form factors for arbitrary shapes',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_modules=[Extension('CDEF.debyer',
        ['src/debyer/atomtables.c', 
        'src/debyer/debyer.c', 
        'src/debyer/debyer_wrap.c', 
        'src/debyer/polyhedrongeom.c'], 
        include_dirs=[numpyinclude])],
    cmdclass={'build_ext':build_ext},
    zip_safe=False,
)
