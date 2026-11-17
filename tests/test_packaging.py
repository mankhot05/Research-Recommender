"""Checks that pyproject.toml and requirements.txt list the same dependencies."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _name(requirement: str) -> str:
    """The normalized package name of a requirement like 'pinecone>=10'."""
    name = re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements_txt() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return {_name(line) for line in lines if line.strip() and not line.startswith("#")}


def _pyproject() -> tuple[set[str], set[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    runtime = {_name(r) for r in project["dependencies"]}
    dev = {_name(r) for r in project["optional-dependencies"]["dev"]}
    return runtime, dev


def test_requirements_txt_matches_pyproject():
    runtime, dev = _pyproject()
    requirements = _requirements_txt()

    assert runtime - requirements == set(), "missing from requirements.txt"
    assert requirements - (runtime | dev) == set(), "missing from pyproject.toml"
