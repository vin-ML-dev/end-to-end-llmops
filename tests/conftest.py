"""Shared test fixtures.

The gateway's require_api_key reads DOMAINBOT_API_KEY from the environment at
REQUEST time. If a developer has exported that key (e.g. from a .env for local
serving), every validation test would get a 401 before the validation logic runs.

This autouse fixture clears the gateway key before each test so tests are
independent of the developer's shell. Tests that WANT auth on set it explicitly
via monkeypatch (those run after this fixture, so their setenv wins).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def clear_gateway_api_key():
    saved = os.environ.pop("DOMAINBOT_API_KEY", None)
    yield
    if saved is not None:
        os.environ["DOMAINBOT_API_KEY"] = saved
