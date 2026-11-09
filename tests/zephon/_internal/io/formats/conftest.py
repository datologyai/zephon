# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_array_record() -> Callable[..., None]:
    module = pytest.importorskip("array_record.python.array_record_module")

    def write(path: Path, records: list[bytes], options: str = "group_size:1") -> None:
        writer = module.ArrayRecordWriter(str(path), options)
        for record in records:
            writer.write(record)
        writer.close()

    return write
