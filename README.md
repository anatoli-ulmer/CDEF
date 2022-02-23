CDEF (*C*omputing *D*ebye's scattering formula for *E*xtraordinary *F*ormfactors)
====

This is a Python package to approximately calculate scattering profiles of arbitrarily shaped nanoparticles for small-angle X-ray scattering (SAXS). CDEF generates a quasi-randomly distributed point cloud in the desired particle shape and then applies Debye’s formula to this ensemble of point scatterers to calculate the SAXS pattern. The shape can be loaded from an STL file. CDEF is implemented as an interface to the open source software [Debyer](https://github.com/wojdyr/debyer) for efficient evaluation of Debye's scattering formula. This allows the fitting of measured scattering data within reasonable time on ordinary desktop computers.

The software is described in detail [in this paper](https://arxiv.org/abs/2109.06570).

This program was written at the Physikalisch-Technische Bundesanstalt (PTB) by Jerome Deumer and Christian Gollwitzer (2022) and is distributed under the terms of the GNU General Public License 3.0. 

Usage
======

The usage of CDEF for computing a scattering curve and fitting measured data is demonstrated in [CDEF\_demo.ipynb](CDEF_demo.ipynb)


Installation
============

The recommended way is to install CDEF  via `pip`:

```bash
pip install CDEF --user
```

Drop the `--user` if you want to install the package system-wide. To ensure you use the correct python, invoke pip as
```bash
python3 -m pip install CDEF --user
```

where `python3` should be replaced by your Python 3 command, e.g. `python3` on most Linux installations or `python` on Anaconda for Windows. 

To install from source, go to the directory where you cloned this repository and enter the following command at the command prompt:

```bash
python3 -m pip install . --user
```

System requirements
===================

CDEF should run on most systems with Python 3.6+ and requires a working C compiler compatible with Python. 
In addition, the following Python packages are required:

* NumPy
* SciPy
* Matplotlib

On Linux and macOS, most setups will work with the system C compiler. For Windows, the installation of a compatible C compiler is described [here](https://wiki.python.org/moin/WindowsCompilers).

License
=======
This program was written at the Physikalisch-Technische Bundesanstalt (PTB) by Jerome Deumer and Christian Gollwitzer (2022) and is distributed under the terms of the GNU General Public License 3.0. 

    CDEF is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CDEF is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CDEF.  If not, see <https://www.gnu.org/licenses/>.



