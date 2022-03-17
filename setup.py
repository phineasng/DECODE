"""Install package."""
import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
VCS_REQUIREMENTS = []
REQUIREMENTS = []
try:
    with open("requirements.txt") as fp:
        for requirement in parse_requirements(fp):
            requirement = str(requirement)
            if vcs.search(requirement):
                VCS_REQUIREMENTS.append(requirement)
            else:
                REQUIREMENTS.append(requirement)
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

# TODO: Update these values according to the name of the module.
setup(
    name="DECODE",
    version=read_version("intcr/__init__.py"),  # single place for version
    description="Installable package for DECODE.",
    long_description=open("README.md").read(),
    url="https://github.com/phineasng/DECODE",
    author="An-phi Nguyen, Anna Maria Weber",
    author_email="uye@zurich.ibm.com, wbr@zurich.ibm.com",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"flans": ["py.typed"]},
    #entry_points="""
    #    [console_scripts]
    #    salutation=blueprint.complex_module.core:formal_introduction
    #""",
    scripts=["bin/run_interpretability_pipeline"],
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            "pytest",
            "pytest-cov",
            # checks
            "black==21.5b0",
            "flake8",
            "mypy",
            # docs
            "sphinx",
            "sphinx-autodoc-typehints",
            "better-apidoc",
            "six",
            "sphinx_rtd_theme",
            "myst-parser",
        ],
    },
    install_requires=REQUIREMENTS,
)
