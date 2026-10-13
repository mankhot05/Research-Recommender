"""Smoke test: every app module imports cleanly."""
import importlib

import pytest

MODULES = ["main", "graph", "retrieval", "embeddings", "sources", "llm", "schemas"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"app.{name}")
