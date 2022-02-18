#! /usr/bin/env python3

import os
import sys
import sysconfig
import platform
import subprocess
import tempfile
import shutil

from setuptools import Extension, find_packages
from numpy.distutils.core import setup


def ac_check_flag(flags, script):
    # emulate AC_CHECK_FLAG from autotools to test if the compiler
    # supports a given flag. Return the first working flag. 

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              sysconfig.get_config_var('CC')).split()

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
            with open(os.devnull, 'w') as fnull:
                compiler_result = subprocess.run(compiler + [flag, filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                #print(compiler_result)
                exit_code = compiler_result.returncode
        except OSError :
            exit_code = 1

        # Clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
        
        if exit_code == 0:
            return flag

    return ""


def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
        
    omptestprog = '''
        #include <omp.h>
        #include <stdio.h>
        int main() {
            #pragma omp parallel
            printf("Hello from thread %d, nthreads %d\\n", omp_get_thread_num(), omp_get_num_threads());
        }
'''
    
    ompflags = ['-fopenmp', '/openmp']
    
    ompflag  = ac_check_flag(ompflags, omptestprog);

    if ompflag == "":
        return []

    return [ompflag]


print("Checking for OpenMP support...\t", end="")

extraompflag = check_for_openmp()

print(" ".join(extraompflag))

if extraompflag == []:
    print ("""WARNING
OpenMP support is not available in your default C compiler
The program will only run on a single core. 
""")

setup(
    name='CDEF',
    version='1.0',
    author='Jerome Deumer',
    author_email='jerome.deumer@ptb.de',
    description='Computing form factors for arbitrary shapes',
    long_description='',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_modules=[Extension('debyer',
        ['src/debyer/atomtables.c', 
        'src/debyer/debyer.c', 
        'src/debyer/debyer_wrap.c', 
        'src/debyer/polyhedrongeom.c'], 
        extra_compile_args = extraompflag,
        extra_link_args = extraompflag)],
    zip_safe=False,
)
