#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run checkout tests against wheel modules without pytest replacing package parents."""

import importlib
from pathlib import Path
import sys

import pytest

paths = [Path(p).resolve() for p in sys.argv[1:]]
for path in paths:
    parts = path.parts
    start = parts.index("dimos")
    for end in range(start + 1, len(parts)):
        name = ".".join(parts[start:end])
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        if filename:
            assert "site-packages" in filename, (name, filename)
import dimos

print("Installed production root:", dimos.__file__, flush=True)
raise SystemExit(
    pytest.main(
        ["--noconftest", "--import-mode=importlib", "-o", "addopts=", "-q", *map(str, paths)]
    )
)
