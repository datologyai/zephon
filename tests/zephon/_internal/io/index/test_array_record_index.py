# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""ArrayRecord index metadata extraction through the supplied storage backend."""

from unittest import mock

from zephon._internal.io.index import array_record_index
from zephon._internal.io.index.array_record_index import ArrayRecordIndexBuilder


def test_extract_shard_info_reuses_known_size_and_backend() -> None:
    storage = mock.Mock()
    with mock.patch.object(
        array_record_index, "_read_array_record_row_count", return_value=7
    ) as count:
        info = ArrayRecordIndexBuilder(storage).extract_shard_info(
            "custom://root/a.array_record", 123
        )
    count.assert_called_once_with("custom://root/a.array_record", storage, 123)
    assert info.basename == "a.array_record"
    assert info.bytes == 123
    assert info.num_rows == 7
    assert info.extra == {"length": 7}
    storage.stat.assert_not_called()
    storage.download.assert_not_called()
