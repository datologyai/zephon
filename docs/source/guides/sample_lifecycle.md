# Sample lifecycle, determinism, and replay

This note describes the shipped behavior: how samples flow through Zephon, how we guarantee determinism and checkpoint/replay, how eviction/tombstones work, and what contracts new operators must follow (fan‑out, map, packing, shuffle, filtering).

## Vocabulary and identities
- **Lane**: Logical replica of the pipeline. Each lane owns its own `WorkSource`, chunk numbering, inflight chunks, and replay sentinel.
- **Chunk**: A `WorkChunk` emitted by a lane’s `WorkSource`. Chunk ids (`chunk_id`) are monotonically increasing per lane and are persisted.
- **Base sample / base offset**: A single element inside a chunk, addressed by `(chunk_id, chunk_offset)`.
- **SampleId**: `(dataset_id, shard_id, local_id)` provided by the `WorkSource`.
- **SampleMeta / SampleCursor**: `SampleMeta` carries `sample_id`, `lane_id`, `chunk_id`, `chunk_offset`, `lineage`, and `tags`. The `meta.contributors` and `meta.tombstone` properties live inside `tags` under `_contributors` and `_tombstone`. `meta.cursor` (ordered by chunk_id → chunk_offset → lineage → sample_id) is the unique identity of the **record being emitted**.
- **Lineage / children**: Fan‑out must derive children via deterministic indices so cursors stay unique and deterministic per lane. Children can have children; lineage simply appends indices.
- **Contributor**: A contributor describes which base offset a record depends on, and whether that record closes the base offset. Represented as `ContributorRef(cursor, is_last_child)`.
- **Packed record**: A record whose payload combines contributors from multiple base offsets. Its `meta.cursor` is the record identity for replay; `meta.contributors` lists every `ContributorRef` included, with `is_last_child` set where that packed record completes a base offset.
- **Tombstone**: A record whose only purpose is to close a base offset when no real payload is emitted. `tombstone_meta(ref, lane_id)` marks `meta.tombstone` (via the `_tombstone` tag) and provides a closing `ContributorRef`. The pipeline notifies the engine about it but does not deliver it to the training consumer.

Helper builders (in `zephon/core/children.py`):
- `spawn_child(parent, child_idx, is_last_child=False, tags=None)` – deterministic fan‑out that sets both lineage and contributor metadata. If the parent already has contributors (e.g., packed input), they are propagated; `is_last_child=True` marks all inherited contributors as closing.
- `pack_meta(primary_cursor, contributors, lane_id, tags=None)` – packed outputs.
- `tombstone_meta(ref, lane_id)` – close an offset without payload.

## End-to-end lifecycle
1. **WorkSource → chunks**: Each lane owns a `WorkSource`. `_lane_stream` replays restored inflight chunks in ascending `chunk_id`, then fetches fresh chunks, assigning monotonically increasing ids per lane. Within a chunk, record order is whatever the `WorkChunk` deterministically provides.
2. **Operators → children/contributors**: Fan‑out uses deterministic `child_idx` lineage. When a base sample yields multiple outputs, use `spawn_child(..., is_last_child=...)` so exactly one output per base offset closes it. Map-style ops that stay 1:1 can leave `contributors` empty to get the default single-contributor semantics.
3. **Tail multiplexing**: If a worker owns multiple lanes, `_lane_rr_iter` emits one item at a time per lane in round‑robin order keyed by the current physical topology. This is a fairness hint; correctness comes from per-lane state, not the RR pointer.
4. **Notify and emit**: `Pipeline._yield_while_notifying` forwards each tail item after notifying the engine. When the plan advertises `preserves_cursor_order=True` (no cross-chunk reordering/packing), it uses the simple chunk-watermark path; otherwise it forwards contributor refs and the record-level cursor. Tombstones are notified but not yielded to the user/training loop.
5. **Checkpoint/replay**: `Engine.state_dict()` persists inflight chunks, per-lane pointers, work source state, RR hints, and the per-lane replay sentinel. On restore, inflight chunks are replayed deterministically; replay filtering drops already-consumed records by equality.

## Determinism building blocks
- **Per-lane chunk order**: Restored inflight chunks are replayed in ascending `chunk_id`; new chunks get strictly increasing ids per lane. Chunk contents are read deterministically via `WorkChunk`.
- **Lane RR fairness**: `_lane_rr_iter` uses an RR pointer keyed by `rank:worker/active:owned_lanes`. Topology changes recompute the pointer from progress; durable correctness is preserved by per-lane state.
- **Single-lane semantics**: Runners enforce single-threaded semantics per lane. Child lineage indices must follow emission order so `SampleCursor` matches that deterministic order.
- **Packed/shuffled output**: Operators may reorder outputs, pack across chunks, or shuffle, but must be deterministic given identical inputs and must keep `meta.cursor` unique per lane for each emitted record.
- **Lane ownership**: Lanes are partitioned deterministically across ranks/workers; idle workers own no lanes. Progress is stored per logical lane, so remapping lanes to different workers/ranks across checkpoints is safe.

## Progress tracking, eviction, and state management
`Pipeline._yield_while_notifying` extracts progress in one of two ways:
- **Cursor-ordered plan (`preserves_cursor_order=True`)**: Uses the simple chunk-watermark path. For each `SampleRecord` or records inside a `SampleBatch`, it computes `max_chunk_id` and the cursors from that chunk, calls `engine.notify_monotone(...)`, and relies on the assumption that tail emission is monotone in `SampleCursor` (no cross-chunk reordering/packing). No per-offset bitmap is maintained in this mode.
- **General plan (`preserves_cursor_order=False`)**: Uses contributor-level progress. For each `SampleRecord` or record inside a `SampleBatch`, it gathers `meta.contribution_refs()` (defaults to the record’s own cursor with `is_last_child=True` when `contributors` is empty) and a `record_cursor` (the record identity for replay; for batches this is the last record’s cursor). Tombstones are notified and then dropped from the user-visible stream. This path maintains per-chunk offset bitmaps to safely evict chunks when every base offset has closed.

`Engine.notify(lane_id, entries, record_cursor)` updates:
- **Per-offset completion (general path)**: `_offset_done[lane][chunk_id]` is a bitset of offsets that have seen `is_last_child=True`. `_offset_done_count` tracks how many offsets are closed per chunk. These are runtime-only (not checkpointed).
- **Eviction**:
  - **Simple path** (`preserves_cursor_order=True`): evicts all inflight chunks with `cid < max_chunk_id` because tail emission is monotone and each output has a single contributor.
  - **Epoch-based path** (`preserves_cursor_order=False`): chunks are grouped into epochs by flush sentinels (injected every `flush_every_k_chunks` chunks per lane). An epoch is evicted atomically once all offsets in all its chunks are closed. Before the first sentinel fires, a fallback "all below epoch floor" check enables eviction of early chunks (important for thread/process runners where the pump runs ahead). The replay cursor's chunk is pinned — its epoch is never evicted until the cursor advances past it.
- **Lane progress**: `_lane_progress[lane] = LanePtr(front_chunk, completed_offsets_in_front_chunk)` for fairness diagnostics and RR seeding.
- **Replay sentinel**: `_lane_last_cursor[lane] = record_cursor` (last delivered record for that lane).

State persisted in `state_dict()` (one shard per worker, merged across ranks):
- Inflight chunks per lane (`WorkChunk.state_dict()`), lane progress (`LanePtr`), next chunk id per lane, `WorkSource` state per lane and global, RR pointer map, last round metadata, checkpoint reload count, and **one replay cursor per lane** (`SampleCursorKey` or `None`).
- **Epoch boundaries** (non-monotonic only): list of sentinel `chunk_id`s per lane marking flush points.  This can include a trailing boundary just beyond the last inflight chunk, preserving the flush point between replayed data and newly fetched data.  On restore, sentinels are re-injected at these positions during Phase 1 so accumulators flush at the same points as the original run.
- **Not persisted**: `_offset_done` bitmaps and counts; they are rebuilt by deterministic replay of inflight chunks after restore.

On `load_state_dict()`:
- Lane work sources are re-cloned, inflight chunks restored, lane pointers and next chunk ids reinstated, RR pointer refreshed from progress, runtime offset trackers cleared, and `_lane_last_cursor` loaded (or cleared if replay disabled). `_publish_replay_snapshot()` only publishes a replay target if the cursor’s chunk is still inflight; otherwise `None` is advertised because that record will not reappear. Offset bitmaps are rebuilt by streaming the restored inflight chunks; this keeps checkpoints small.

### Contributor and tombstone invariants
- For each base offset `(chunk_id, chunk_offset)` and lane, **exactly one** contributor or tombstone across the whole pipeline must have `is_last_child=True`.
- If multiple outputs are emitted for a base offset, mark exactly one with `is_last_child=True` (after all others for that offset have been emitted or dropped).
- If every real output is dropped, emit a tombstone carrying a `ContributorRef(..., is_last_child=True)` so the engine can close that offset and eventually evict the chunk.
- `meta.contributors` should enumerate all contributors whose content is inside a record. Contributors that close offsets must set `is_last_child=True`; others set it to `False`.
- `meta.cursor` is the identity of the *record* for replay. Contributors capture which base offsets that record depends on for eviction.
- A packed record may carry multiple contributors and can close multiple offsets (across chunks) at once.
- **Safe default rule**: Operators that drop items SHOULD always emit tombstones, regardless of `preserves_cursor_order`. An operator cannot know at build time which notify path the plan will use (it depends on the AND of all operators' traits). Tombstones are harmless in the monotone path (notified then skipped) and required in the general path (close per-offset bitmaps). Always emitting tombstones is correct in all configurations.

### Eviction details and why it is safe
- **Monotone path assumptions**: Tail emission per lane is monotone in `SampleCursor` and each output has exactly one contributor (no packing/tombstones). Under these assumptions, seeing chunk `N` implies no future outputs will reference chunks `< N`, so evicting `cid < N` is safe and no bitmaps are required.
- **Epoch-based path**: Safe eviction requires two conditions: (1) all offsets in the chunk are closed (per-offset bitmaps), and (2) no accumulator state influenced by the chunk affects future output. Condition 2 is guaranteed by epoch boundaries — flush sentinels reset history-dependent accumulators, so cross-epoch eviction is safe. Chunks within an epoch are evicted atomically.
- **Cursor pinning**: the replay cursor's chunk is never evicted, even if both conditions are met, to ensure the ReplayFilter can find its target on resume.
- **Before first sentinel**: a fallback “all below epoch floor” atomic check enables eviction before any epoch boundary exists. This matters for thread/process runners where the pump runs ahead of the consumer.
- **Replay/checkpoint**: only inflight chunks (plus epoch boundary positions) are persisted. Evicted chunks never replay; on restore, inflight chunks re-stream their contributors/tombstones to rebuild the bitmaps, and sentinels are re-injected at stored positions, so eviction state is recovered deterministically.
- **Safety**: forgetting a closing contributor/tombstone for an offset prevents eviction of that chunk (and later ones) instead of silently dropping needed data — memory pressure will surface the bug.

## Replay and checkpoint/restart behavior
- **Stored state per lane**: The last delivered record cursor (a sentinel), plus inflight chunks, lane pointers, and epoch boundary positions (non-monotonic only). No high-watermark math is used.
- **ReplayFilter** (auto-inserted before batching or at the tail) drops records until it sees the sentinel again (inclusive), then emits the suffix. Targets initialize to `None` (emit everything) if no prior progress.
- **Non-monotonic safe**: Equality-based replay works even if tail emission order is non-monotonic in `SampleCursor` (e.g., cross-chunk shuffle or packing).
- **Evicted sentinels**: If the sentinel’s chunk is already evicted at checkpoint time, `_publish_replay_snapshot()` sets the replay target to `None` because that record will not reappear; emitting everything on resume is then correct.
- **Multi-lane independence**: Each lane replays independently. Cross-lane interleave may differ after remap, but each lane’s suffix matches the crash-free run.
- **Batches**: For a `SampleBatch`, the replay sentinel is the last record’s cursor; the batch containing the checkpoint boundary is dropped entirely on resume.
- **Tombstones**: Participate in eviction but are removed from the user stream before training; replay is keyed on record-level cursors of real records.

## Contracts for operators (fan-out, map, packing, shuffle)
- **How to think about contributors vs children**:
  - **Children/lineage**: Any fan-out must still use deterministic `child_idx` to derive `meta.lineage` so each emitted record has a unique `meta.cursor`.
  - **Contributors**: The eviction accounting unit. By default every record is treated as a single contributor with `is_last_child=True`. This is only correct when a base sample produces **exactly one** output record.
  - **When to use `spawn_child`**: Whenever a base sample yields multiple output records (any true fan-out), call `spawn_child(parent, idx, is_last_child=...)` for *every* emitted record. Do not re-emit the parent metadata unchanged; that would duplicate cursors. Exactly one child per base offset should set `is_last_child=True` (the last one you emit for that base). If the parent already has contributors (e.g., packed input), `spawn_child` propagates them and optionally marks them closed.
  - **Packing view**: A packer sees its inputs as contributors and emits one packed record with its own `meta.cursor` while listing the contributors it included. Those contributors might be straight-through records (default contributor when 1:1) or explicit contributors built via `spawn_child` for multi-output fan-out.
- **General**:
  - Keep emission deterministic given the same inputs; seed RNGs if used.
  - Preserve `sample_id`, `lane_id`, `chunk_id`, and `chunk_offset` from the base input unless intentionally creating a new primary cursor for a packed record.
  - Ensure `meta.cursor` is unique per emitted record per lane; replay identity depends on it reappearing.
- **Fan-out / contributor producers**:
  - Use `spawn_child(parent, idx, is_last_child=...)` when a single base sample produces multiple outputs so lineage matches emission order and you can mark the one that closes the base offset. If you emit exactly one record per base sample, the default single-contributor behavior is sufficient.
  - Exactly one contributor per base offset (or tombstone) must set `is_last_child=True`. If all real outputs are dropped, emit a tombstone closing that offset.
- **Packing operators**:
  - Choose a deterministic `primary_cursor` for the packed record (often one contributor's cursor plus an extra lineage step if needed) and ensure uniqueness per lane.
  - Build `contributors` for every included contributor; set `is_last_child=True` on contributors that close their base offsets. Use `pack_meta(...)` to assemble the metadata.
  - A single packed record may close multiple offsets across chunks; the engine will evict chunks once all offsets are closed.
  - Example (see `PackSequences._create_packed_record` for the canonical implementation):
    ```python
    from zephon.core.children import pack_meta
    contributors = []
    for sample in samples:
        contributors.extend(sample.meta.contribution_refs())
    primary_cursor = samples[0].meta.cursor.child(0)
    packed_meta = pack_meta(
        primary_cursor=primary_cursor,
        contributors=contributors,
        lane_id=samples[0].meta.lane_id,
        component_sample_counts=aggregated_counts,
    )
    packed_record = SampleRecord(meta=packed_meta, payload=packed_payload)
    ```
- **Filtering / dropping**:
  - If dropping the final contributor for an offset (or dropping all contributors), emit `tombstone_meta(ref, lane_id)` so eviction can progress. Batching forwards tombstones without affecting batch shapes; the pipeline hides them from the training consumer.
  - `MapTransform` handles this automatically when `drop_none=True`. For custom operators, iterate `contribution_refs()` and emit a tombstone for each ref with `is_last_child=True`.
- **Stateful transforms** (`Pipeline.stateful_transform()`):
  - Set `preserves_cursor_order=False` when your push/transform function reorders items (shuffle) or packs multiple records into one. The default is `True`, which selects the monotone notify path. If your transform actually reorders, this can lead to premature chunk eviction.
  - If your transform drops items (e.g., dedup filtering in `push_fn` or length filtering in `transform_fn`), those drops happen inside the operator and do NOT automatically emit tombstones. You must track which items were dropped and emit tombstone records from `transform_fn` or `push_fn`. Alternatively, perform filtering in a preceding `MapTransform` with `drop_none=True` which handles tombstones automatically.
  - If your transform buffers across chunk boundaries and cannot meaningfully flush mid-stream, note that Zephon does not currently support intentional stalling for custom stateful transforms. Only `Batch(drop_last=True)` uses stalling today; general stalled operators would need additional replay-capsule support.
- **Map-style transforms**:
  - If you only mutate payloads and keep a 1:1 mapping, reuse the incoming `SampleMeta`. The default `contribution_refs()` handles eviction/replay correctly.
  - `MapTransform` with `drop_none=True` automatically emits tombstones for every closing contributor in a dropped item (both `SampleRecord` and `SampleBatch`). No manual tombstone handling is needed when using the `Pipeline.map_transform()` API.
  - Custom operators that conditionally drop records must emit tombstones for every closing contributor (`is_last_child=True`) in the dropped item's `contribution_refs()`. See `MapTransform._tombstones_for()` for reference.
- **Flush contract** (for operators with `preserves_cursor_order=False`):
  - `flush(reset=True)` must emit all buffered data and **fully reset** internal state so the accumulator is indistinguishable from a freshly constructed instance. This is called at epoch boundaries (flush sentinels) to guarantee clean replay.
  - `flush()` (default `reset=False`) is called at end of stream. Semantics are operator-defined (e.g., `Batch` with `drop_last=True` discards partial batches).
  - The only supported intentional stalling case is `Batch(drop_last=True)`. For general operators, relying on `stall_on_epoch_boundary=True` is not supported today because replay after eviction would need extra cross-boundary state beyond the current checkpoint payload.
  - For non-Batch operators, `has_pending_data()` must be `False` after `flush(reset=True)`. If pending data remains, Zephon treats that as a contract violation and raises.
- **Shuffle/cross-chunk buffering**:
  - Reordering is allowed; replay remains correct. Still respect the contributor/tombstone contract so eviction can safely remove chunks whose offsets are fully closed.
- **Closing contributors after reordering/packing**:
  - Every base offset must still have exactly one outgoing contributor (or tombstone) with `is_last_child=True`, even if you reshuffle or pack records.
  - If you reorder without packing (e.g., a shuffle buffer), track the last emitted record *per base offset* and move `is_last_child=True` onto that record’s contributor(s); when every base offset appears only once in the batch, no rewrite is needed.
  - If you pack, gather contributors for all inputs included in the pack. For any base offset that appears multiple times in the packed inputs, carry **at most one** contributor in the packed record; mark it closing if the last child for that base offset is inside the pack.
  - When dropping an input that was the last child for a base offset, emit a tombstone to keep eviction unblocked; batching and packing must forward tombstones unchanged.
  - Helpers:
    - `SampleCursor.base_offset` exposes the `(chunk_id, chunk_offset)` pair used for eviction bookkeeping; use it instead of rolling your own tuple construction.
    - Operators that only reorder (no drops/inserts) can reuse a shared helper to rewrite `is_last_child` onto the last occurrence per base offset; the helper must not be used when dropping closers (emit tombstones instead).  Currently lives in the shuffle buffer implementation.
- **Avoiding pitfalls**:
  - Do not generate multiple `is_last_child=True` contributors for the same base offset.
  - Do not pick a non-deterministic `primary_cursor` for packed outputs.
  - Do not drop tombstones before engine notification; the pipeline already removes them from the user stream after notifying.

## Practical walkthrough
1. WorkSource hands lane 0 chunk 5 (cid=5). Engine assigns cid=5, keeps it inflight, yields records `(sample_id, lane_id=0, cid=5, offset=...)`.
2. A splitter emits two children for offset 3 using `spawn_child(..., is_last_child=True)` on the second child. A packer later combines that child with another offset and marks both contributors that close their offsets with `is_last_child=True`.
3. Pipeline notifies the engine with those contributors; the bitset for chunk 5 marks offset 3 as done. Once all offsets in chunk 5 are closed (including any tombstones for dropped samples), engine evicts chunk 5.
4. Checkpoint stores inflight chunks, lane pointers, lane_next_cid, work source state, RR state, and the last record cursor per lane. Offset bitmaps are rebuilt on restore by reprocessing the saved inflight chunks.
5. After restart, ReplayFilter drops records until it sees the saved cursor per lane, then emits the suffix exactly once; eviction proceeds identically because the contributor/tombstone signals are replayed.

### Example: cross-chunk packing with a filtered-out base
- Lane 0 has chunk 10 offsets 0,1 and chunk 11 offsets 0,1.
- Operator splits `(10,0)` into two children, marking the second as `is_last_child=True`. `(10,1)` yields one child marked last. `(11,1)` yields one child marked last. `(11,0)` is discarded by a filter.
- Packer creates `R0 = pack(f10_0a, f10_0b)` with contributors: `f10_0a` (`is_last_child=False`), `f10_0b` (`True`, closes (10,0)).
- Packer creates `R1 = pack(f10_1, f11_1)` with contributors both marked `True` (closes (10,1) and (11,1)).
- Filter emits a tombstone for `(11,0)` using `tombstone_meta(ref_for_11_0, lane_id=0)`.
- Engine sees closures for all offsets in chunks 10 and 11; evicts both in order. Replay stores only the record-level last cursor; suffix replay works even though emission order is non-monotonic in cursor space.

### Example: dedup filter dropping a last contributor
- Suppose `(12,4)` would produce a single child marked `is_last_child=True`, but a dedup filter decides to drop it.
- Without action, chunk 12 could never evict because `(12,4)` never closes.
- The filter emits `tombstone_meta(ContributorRef(cursor_of_12_4, is_last_child=True), lane_id)`. The engine marks offset 4 complete; batching passes the tombstone through without affecting shapes; the pipeline hides it from the training loop. Eviction remains correct and bounded.
