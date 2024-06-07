# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open
from os import path

import numpy as np
import setuptools as st
from Cython.Build import cythonize

ext_modules = cythonize(
    [
        "sebbs/change_detection.pyx",
    ]
)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt")) as f:
    requirements = list(filter(lambda x: len(x) > 0 and not x.startswith("#"), f.read().split("\n")))

st.setup(
    name="sebbs",
    version="0.0.0",
    description="Sound Event Bounding Boxes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/merlresearch/sound_event_bounding_boxes",
    author="Mitsubishi Electric Research Laboratories",
    author_email="ebbers[at]merl[dot]com",
    license="AGPL-3.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: AGPL-3.0 License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="sound event detection, polyphonic sound detection, post processing, change detection",
    packages=st.find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pathlib",
        "sed_scores_eval",
        "Cython",
    ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage", "jupyter", "matplotlib"],
    },
    ext_modules=ext_modules,
    package_data={"sebbs": ["**/*.pyx"]},  # https://stackoverflow.com/a/60751886
    include_dirs=[np.get_include()],
)
