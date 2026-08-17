# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Node-shared decoded Parquet row-group cache coordinator."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zephon._internal.io.formats.arrow_rows import take_and_materialize
from zephon._internal.io.formats.parquet_cache.admission import (
    ParquetRGAdmission,
    RGAdmissionOutcome,
    RGEvictionOutcome,
    RGPublicationResult,
)
from zephon._internal.io.formats.parquet_cache.codec import ArrowRGFileCodec
from zephon._internal.io.formats.parquet_cache.control import DecodedRGState
from zephon._internal.io.formats.parquet_cache.index import ParquetRGIndex
from zephon._internal.io.formats.parquet_cache.session import ParquetRGSession
from zephon._internal.io.ofd_lock import OFDLease

logger = logging.getLogger(__name__)

_DIRECTORY_SHARD_SIZE = 4096
_SAME_RG_WAIT_SECONDS = 0.100
_METADATA_WAIT_SECONDS = 0.025
_EVICTION_BUDGET_SECONDS = 0.100
_CLOCK_CHUNK_SLOTS = 64
_MAX_CLOCK_SLOTS = 4096
_THRASH_MIN_PUBLICATIONS_AFTER_FILL = 256
_THRASH_RATE = 0.10
_THRASH_WARNING_INTERVAL_SECONDS = 30.0


class ParquetRGCache:
    """Coordinate the decoded row-group cache for one process.

    Readers call ``get_or_decode``; this object combines ``ParquetRGIndex``,
    session/admission coordination, and ``ArrowRGFileCodec`` to read or publish
    node-shared files while always returning caller-owned rows.

    Callers encountering a cold row group already being built wait briefly for
    its publication, then decode directly instead of joining an unbounded wait;
    only the elected builder may publish the shared cache file.
    """

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        index: ParquetRGIndex,
        limit_bytes: int,
        min_free_bytes: int,
    ) -> None:
        if index.num_row_groups <= 0:
            raise ValueError("Decoded RG cache requires at least one Parquet row group")
        self._index = index
        self._min_free_bytes = min_free_bytes
        self._codec = ArrowRGFileCodec()
        configuration = self._configuration_fingerprint(
            index=index,
            limit_bytes=limit_bytes,
            min_free_bytes=min_free_bytes,
        )
        self._session = ParquetRGSession(
            root,
            slot_count=index.num_row_groups,
            limit_bytes=limit_bytes,
            catalog_fingerprint=index.fingerprint,
            configuration_fingerprint=configuration,
        )
        self.root = self._session.root
        self._admission = ParquetRGAdmission(
            self._session,
            discard_paths=self._discard_paths,
        )
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "builds": 0,
            "bypasses": 0,
            "publications": 0,
            "evictions": 0,
            "reloads": 0,
            "corruptions": 0,
        }
        self._thrash_warning_window: tuple[float, int, int] | None = None
        self._closed = False

    def get_or_decode(
        self,
        *,
        slot: int,
        local_indices: list[int],
        decode: Callable[[], Any],
    ) -> list[dict[str, object]]:
        """Read a mapped hit or decode once, always returning owned rows."""
        self._check_usable()
        if not local_indices:
            return []

        hit = self._try_hit(slot, local_indices)
        if hit is not None:
            self._increment("hits")
            return hit
        self._increment("misses")

        exact = self._admission.try_exact_exclusive(slot)
        if exact is None:
            shared = self._admission.wait_ready_shared(
                slot,
                timeout=_SAME_RG_WAIT_SECONDS,
            )
            if shared is not None:
                try:
                    rows = self._read_under_lease(slot, local_indices)
                except Exception:
                    rows = None
                finally:
                    shared.close()
                if rows is not None:
                    self._increment("hits")
                    return rows
            self._increment("bypasses")
            return take_and_materialize(decode(), local_indices)

        with exact:
            try:
                if not self._admission.ensure_clean_for_exact(
                    slot=slot,
                    exact_lease=exact,
                    timeout=_METADATA_WAIT_SECONDS,
                ):
                    self._increment("bypasses")
                    return take_and_materialize(decode(), local_indices)
                self._recover_stale_slot(slot, exact)
            except Exception as exc:
                self._log_cache_bypass("recover", slot, exc)
                self._increment("bypasses")
                return take_and_materialize(decode(), local_indices)
            if self._session.control.slot(slot).state is DecodedRGState.READY:
                try:
                    rows = self._read_under_lease(slot, local_indices)
                except Exception as exc:
                    self._log_cache_bypass("read", slot, exc)
                    with contextlib.suppress(Exception):
                        self._discard_ready(slot, exact, error=exc)
                else:
                    self._increment("hits")
                    return rows

            table = decode()
            self._increment("builds")
            rows = take_and_materialize(table, local_indices)
            self._try_publish(slot, exact, table)
            return rows

    def stats(self) -> dict[str, int]:
        """Return process-local outcomes plus node-shared byte counters."""
        with self._stats_lock:
            stats = dict(self._stats)
            thrash_warned = int(self._thrash_warning_window is not None)
        control = self._session.control
        header = control.header()
        ready_count = 0
        if not header.dirty and not header.disabled:
            ready_count = control.count_slots(DecodedRGState.READY)
        stats.update(
            {
                "accounted_bytes": header.accounted_bytes,
                "inflight_bytes": header.inflight_bytes,
                "ready_count": ready_count,
                "thrash_warned": thrash_warned,
                "node_post_fill_publications": header.post_fill_publications,
                "node_post_fill_reloads": header.post_fill_reloads,
            }
        )
        return stats

    def close(self) -> None:
        """Release this process's session attachment; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        self._session.close()

    def _try_hit(
        self,
        slot: int,
        local_indices: list[int],
    ) -> list[dict[str, object]] | None:
        lease = self._admission.try_ready_shared(slot)
        if lease is None:
            return None
        try:
            return self._read_under_lease(slot, local_indices)
        except Exception as exc:
            read_error = exc
            self._log_cache_bypass("read", slot, exc)
        finally:
            lease.close()
        exact = self._admission.try_exact_exclusive(slot)
        if exact is not None:
            with exact:
                with contextlib.suppress(Exception):
                    if self._admission.ensure_clean_for_exact(
                        slot=slot,
                        exact_lease=exact,
                        timeout=_METADATA_WAIT_SECONDS,
                    ):
                        self._discard_ready(slot, exact, error=read_error)
        return None

    def _read_under_lease(
        self,
        slot: int,
        local_indices: list[int],
    ) -> list[dict[str, object]]:
        snapshot = self._session.control.slot(slot)
        if snapshot.state is not DecodedRGState.READY:
            raise RuntimeError(f"Decoded RG slot {slot} is no longer READY")
        with self._codec.open_mapped(
            self._final_path(slot),
            self._index.identity_for(slot),
            exact_bytes=snapshot.bytes,
        ) as mapped:
            return mapped.take_and_materialize(local_indices)

    def _try_publish(self, slot: int, exact: OFDLease, table: Any) -> None:
        identity = self._index.identity_for(slot)
        try:
            payload_bytes = self._codec.measure(table, identity)
        except Exception as exc:
            self._log_cache_bypass("measure", slot, exc)
            self._increment("bypasses")
            return

        try:
            outcome = self._admission.try_reserve(
                slot=slot,
                exact_lease=exact,
                payload_bytes=payload_bytes,
                min_free_bytes=self._min_free_bytes,
                timeout=_METADATA_WAIT_SECONDS,
            )
            if outcome is RGAdmissionOutcome.PRESSURE:
                outcome = self._reserve_under_pressure(
                    slot=slot,
                    exact=exact,
                    payload_bytes=payload_bytes,
                )
        except Exception as exc:
            self._log_cache_bypass("admission", slot, exc)
            self._increment("bypasses")
            return
        if outcome is not RGAdmissionOutcome.ADMITTED:
            self._increment("bypasses")
            return

        final_path = self._final_path(slot)
        temp_path = self._temp_path(slot)
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            self._codec.write(
                table,
                identity,
                temp_path,
                exact_bytes=payload_bytes,
            )
            os.replace(temp_path, final_path)
            publication = self._admission.publish_ready(
                slot=slot,
                exact_lease=exact,
                timeout=_METADATA_WAIT_SECONDS,
            )
            if publication is not None:
                self._increment("publications")
                self._maybe_warn_thrashing(publication)
            else:
                self._cleanup_failed_build(slot, exact)
        except Exception as exc:
            self._log_cache_bypass("publish", slot, exc)
            self._cleanup_failed_build(slot, exact)

    def _reserve_under_pressure(
        self,
        *,
        slot: int,
        exact: OFDLease,
        payload_bytes: int,
    ) -> RGAdmissionOutcome:
        leader = self._admission.try_eviction_leader()
        if leader is None:
            return RGAdmissionOutcome.CONTENDED
        deadline = time.monotonic() + _EVICTION_BUDGET_SECONDS
        max_scanned = min(self._session.control.slot_count * 2, _MAX_CLOCK_SLOTS)
        scanned = 0
        with leader:
            # A CONTENDED return below may leave this set. That is intentional:
            # a later eviction leader treats the flag as stale and clears it.
            if not self._admission.set_eviction_pending(
                leader_lease=leader,
                pending=True,
                timeout=self._remaining_metadata_timeout(deadline),
            ):
                return RGAdmissionOutcome.CONTENDED

            def try_reserve_again(
                *, clear_pressure: bool = False
            ) -> RGAdmissionOutcome:
                outcome = self._admission.try_reserve_as_leader(
                    slot=slot,
                    exact_lease=exact,
                    leader_lease=leader,
                    payload_bytes=payload_bytes,
                    min_free_bytes=self._min_free_bytes,
                    timeout=self._remaining_metadata_timeout(deadline),
                )
                if outcome is RGAdmissionOutcome.ADMITTED:
                    return outcome
                if clear_pressure or outcome is not RGAdmissionOutcome.PRESSURE:
                    self._admission.set_eviction_pending(
                        leader_lease=leader,
                        pending=False,
                        timeout=self._remaining_metadata_timeout(deadline),
                    )
                return outcome

            while scanned < max_scanned and time.monotonic() < deadline:
                outcome = try_reserve_again()
                if outcome is not RGAdmissionOutcome.PRESSURE:
                    return outcome
                chunk = self._admission.claim_clock_chunk(
                    leader_lease=leader,
                    max_slots=min(_CLOCK_CHUNK_SLOTS, max_scanned - scanned),
                    timeout=self._remaining_metadata_timeout(deadline),
                )
                if chunk is None:
                    return RGAdmissionOutcome.CONTENDED
                scanned += len(chunk)
                for victim_slot in chunk:
                    if time.monotonic() >= deadline:
                        break
                    if not self._evict_candidate(victim_slot):
                        continue
                    # Recheck after every actual eviction. A CLOCK chunk is a
                    # bounded scan unit, not an instruction to discard every
                    # eligible member; usually one victim creates enough room.
                    outcome = try_reserve_again()
                    if outcome is not RGAdmissionOutcome.PRESSURE:
                        return outcome

            return try_reserve_again(clear_pressure=True)

    def _evict_candidate(self, slot: int) -> bool:
        exact = self._admission.try_exact_exclusive(slot)
        if exact is None:
            return False
        with exact:
            if not self._admission.ensure_clean_for_exact(
                slot=slot,
                exact_lease=exact,
                timeout=_METADATA_WAIT_SECONDS,
            ):
                return False
            snapshot = self._session.control.slot(slot)
            if snapshot.state is DecodedRGState.BUILDING:
                self._cleanup_failed_build(slot, exact)
                return False
            if snapshot.state is DecodedRGState.EVICTING:
                return self._finish_payload_eviction(slot, exact)
            outcome = self._admission.try_mark_evicting(
                slot=slot,
                exact_lease=exact,
                timeout=_METADATA_WAIT_SECONDS,
            )
            if outcome is RGEvictionOutcome.EVICTING:
                return self._finish_payload_eviction(slot, exact)
            return False

    def _finish_payload_eviction(self, slot: int, exact: OFDLease) -> bool:
        final_path = self._final_path(slot)
        try:
            final_path.unlink(missing_ok=True)
        except OSError:
            self._admission.restore_evicting_ready(
                slot=slot,
                exact_lease=exact,
                timeout=_METADATA_WAIT_SECONDS,
            )
            return False
        if self._admission.finish_eviction(
            slot=slot,
            exact_lease=exact,
            timeout=_METADATA_WAIT_SECONDS,
        ):
            self._increment("evictions")
            return True
        return False

    def _recover_stale_slot(self, slot: int, exact: OFDLease) -> None:
        state = self._session.control.slot(slot).state
        if state is DecodedRGState.BUILDING:
            self._cleanup_failed_build(slot, exact)
        elif state is DecodedRGState.EVICTING:
            self._finish_payload_eviction(slot, exact)
        elif state is DecodedRGState.EMPTY:
            self._discard_paths(slot)

    def _discard_ready(
        self,
        slot: int,
        exact: OFDLease,
        *,
        error: Exception,
    ) -> None:
        # READY slots normally have their refbit set, including by the failed
        # read lease. The first attempt clears that second chance; the second
        # can transition the slot to EVICTING.
        for _ in range(2):
            outcome = self._admission.try_mark_evicting(
                slot=slot,
                exact_lease=exact,
                timeout=_METADATA_WAIT_SECONDS,
            )
            if outcome is RGEvictionOutcome.EVICTING:
                self._increment("corruptions")
                logger.warning(
                    "Discarding unreadable decoded RG cache payload for slot %d: %s",
                    slot,
                    error,
                )
                self._finish_payload_eviction(slot, exact)
                return
            if outcome is RGEvictionOutcome.SKIPPED:
                return

    def _cleanup_failed_build(self, slot: int, exact: OFDLease) -> None:
        try:
            self._temp_path(slot).unlink(missing_ok=True)
            self._final_path(slot).unlink(missing_ok=True)
        except OSError:
            return
        self._admission.release_reservation(
            slot=slot,
            exact_lease=exact,
            timeout=_METADATA_WAIT_SECONDS,
        )

    def _discard_paths(self, slot: int) -> bool:
        """Remove deterministic temp/final names and confirm their absence."""
        for path in (self._temp_path(slot), self._final_path(slot)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return False
        return True

    def _final_path(self, slot: int) -> Path:
        directory = self._session.entries_dir / f"{slot // _DIRECTORY_SHARD_SIZE:06d}"
        return directory / f"{slot:016d}{self._codec.suffix}"

    def _temp_path(self, slot: int) -> Path:
        final_path = self._final_path(slot)
        return final_path.with_name(f".{slot:016d}.tmp")

    def _increment(self, name: str) -> None:
        with self._stats_lock:
            self._stats[name] += 1

    def _maybe_warn_thrashing(self, publication: RGPublicationResult) -> None:
        """Rate-limit warnings to windows containing fresh node-wide churn."""
        now = time.monotonic()
        with self._stats_lock:
            if publication.reloaded_after_eviction:
                self._stats["reloads"] += 1
            node_publications = publication.node_post_fill_publications
            node_reloads = publication.node_post_fill_reloads
            current_window = (now, node_publications, node_reloads)
            window = self._thrash_warning_window
            if window is None:
                warning_publications = node_publications
                warning_reloads = node_reloads
            else:
                started_at, previous_publications, previous_reloads = window
                if now - started_at < _THRASH_WARNING_INTERVAL_SECONDS:
                    return
                warning_publications = node_publications - previous_publications
                warning_reloads = node_reloads - previous_reloads
                if not 0 <= warning_reloads <= warning_publications:
                    # The advisory shared counters started a new generation.
                    self._thrash_warning_window = current_window
                    return

            if warning_publications < _THRASH_MIN_PUBLICATIONS_AFTER_FILL:
                return
            should_warn = warning_reloads >= warning_publications * _THRASH_RATE
            if window is None and not should_warn:
                return
            self._thrash_warning_window = current_window
            if not should_warn:
                return

        logger.warning(
            "Parquet row-group cache is thrashing across this node: %d of %d row "
            "groups decoded into the cache after it filled had previously been "
            "evicted and were re-decoded (%.1f%%; cache cap %.2f GiB). The shuffle "
            "working set exceeds the decoded cache; raise it via "
            "io_options.parquet_rg_cache.limit_bytes to cut redundant decode work.",
            warning_reloads,
            warning_publications,
            min(100.0, 100.0 * warning_reloads / warning_publications),
            self._session.control.limit_bytes / 1024**3,
        )

    def _check_usable(self) -> None:
        if self._closed:
            raise RuntimeError("Decoded RG cache is closed")

    @staticmethod
    def _remaining_metadata_timeout(deadline: float) -> float:
        return max(0.0, min(_METADATA_WAIT_SECONDS, deadline - time.monotonic()))

    @staticmethod
    def _configuration_fingerprint(
        *,
        index: ParquetRGIndex,
        limit_bytes: int,
        min_free_bytes: int,
    ) -> str:
        payload = {
            "decoded_catalog_fingerprint": index.fingerprint,
            "slot_count": index.num_row_groups,
            "limit_bytes": limit_bytes,
            "min_free_bytes": min_free_bytes,
            "payload_codec": "arrow-ipc-file-v1",
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _log_cache_bypass(stage: str, slot: int, error: Exception) -> None:
        logger.debug(
            "Decoded RG cache %s bypass for slot %d: %s",
            stage,
            slot,
            error,
        )

    def __enter__(self) -> "ParquetRGCache":
        self._check_usable()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = ["ParquetRGCache"]
