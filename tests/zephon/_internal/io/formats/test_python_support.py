"""Security-related interpreter restrictions on optional decoders."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from zephon._internal.io.formats import mds


def test_mds_rejects_python310_before_opening_a_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mds, "sys", SimpleNamespace(version_info=(3, 10)))
    reader = Mock()
    monkeypatch.setattr(mds, "_StreamingMDSReader", reader)

    with pytest.raises(RuntimeError, match="MDS decoding requires Python 3.11"):
        mds.MDSFormat().open_shard(Mock(), Mock())

    reader.from_json.assert_not_called()
