"""Shared pytest fixtures and markers for the synlad test suite."""

import os

import pytest

requires_pathway_data = pytest.mark.skipif(
    "TEST_PATHWAY_DATA" not in os.environ,
    reason="TEST_PATHWAY_DATA env var not set; skipping tests that need real pathway pickles",
)
