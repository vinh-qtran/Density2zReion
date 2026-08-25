from __future__ import annotations

import importlib.metadata

import density2zreion as m


def test_version() -> None:
    assert importlib.metadata.version("density2zreion") == m.__version__
