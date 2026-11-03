from __future__ import annotations

import socket
import threading

from zephon._internal.observability.emitter import StatsdEmitter
from zephon.observability.config import MetricsSinkConfig, MetricsSinkMode


def test_batch_flush_does_not_deadlock() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(5.0)
    port = receiver.getsockname()[1]
    emitter = StatsdEmitter(
        MetricsSinkConfig(
            mode=MetricsSinkMode.STATSD,
            statsd_port=port,
            max_batch_size=2,
        )
    )

    def emit() -> None:
        for i in range(3):
            emitter.gauge("g", float(i))

    # Run in a daemon thread so a regression fails the test instead of
    # hanging the whole suite.
    worker = threading.Thread(target=emit, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    try:
        assert not worker.is_alive(), "StatsdEmitter deadlocked on batch flush"

        # Hitting max_batch_size sends the first two gauges as one packet.
        payload = receiver.recv(65536).decode("utf-8")
        assert payload.count("\n") == 1
        assert "zephon.pipeline.g:0" in payload

        emitter.close()
        payload = receiver.recv(65536).decode("utf-8")
        assert "zephon.pipeline.g:2" in payload
    finally:
        receiver.close()
