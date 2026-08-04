"""The two places the version lives must agree.

A package that reports one version and publishes another is a release
defect, and it is the kind that is only ever noticed downstream.
"""

import pathlib
import tomllib

import fz


def test_package_version_matches_pyproject():
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert fz.__version__ == declared
