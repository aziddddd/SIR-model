from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

ext_modules = [
        Extension(
            "ca_utils_cy", 
            ["ca_utils_cy.pyx"],
            include_dirs=[np.get_include()]
        ),
]

setup(
    cmdclass    = {'build_ext': build_ext},
    ext_modules = cythonize(
        ext_modules,
        compiler_directives={'language_level' : "3"}
    )
)