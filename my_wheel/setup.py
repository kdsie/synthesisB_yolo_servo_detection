from setuptools import Extension, setup
import numpy as np
import pybind11

ext_modules = [
    Extension(
        "filter_module",
        ["fliter.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    name="image_filter",
    version="0.1.0",
    author="project team",
    author_email="team@example.com",
    description="A 5x5 image convolution module for performance comparison",
    ext_modules=ext_modules,
    install_requires=["numpy<2", "pybind11"],
    python_requires=">=3.8",
)
