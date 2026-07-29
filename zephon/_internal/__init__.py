# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon internals. This is mostly implementation machinery, not a public API.

Nothing outside the ``zephon`` package should import from ``zephon._internal``.
Its module layout, contents, and signatures may change without notice. Build
and extend pipelines via the public surface (``zephon.Pipeline``,
``zephon.ops``, ``zephon.work``, ``zephon.io``, ``zephon.types``).
"""
