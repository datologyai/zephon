# Record lineage, progress, replay, and packing/shuffle support

This note captures the old (pre #71) lineage/progress model, how fan‑out and filtering work, why packing and cross-chunk buffering/shuffling break today’s assumptions, and the extensions needed to handle both safely intorduced in #71. We keep around this doc for documentation purposes to re-visit the thoughts we had at the time.

## 1) Current model: lineage, progress, eviction, replay (today)

This section describes current behavior, to anchor changes.

### 1.1 Per-record cursor and lineage

`zephon/core/constants.py`:

- `SampleCursor`:

  ```python
  SampleCursorKey = tuple[ChunkId, ChunkOffset, LineagePath, SampleId]

  @dataclass(frozen=True, slots=True)
  class SampleCursor:
      chunk_id: ChunkId
      chunk_offset: ChunkOffset
      sample_id: SampleId
      lineage: LineagePath = field(default_factory=tuple)
      ...
  ```

- Ordering (`__lt__`, `__le__`, etc.) is lexicographic on:

  1. `chunk_id`
  2. `chunk_offset`
  3. `lineage`
  4. `sample_id`

- `SampleMeta.cursor`:

  ```python
  @property
  def cursor(self) -> SampleCursor:
      return SampleCursor(
          self.chunk_id, self.chunk_offset, self.sample_id, self.lineage
      )
  ```

So each record has a unique, deterministic position in `(chunk, offset, lineage, sample_id)` space.

### 1.2 Work, lanes, and chunks

In `Engine._lane_stream`:

- A lane owns its own `WorkSource`.
- Each `next_chunk()` returns a `WorkChunk` for that lane.
- Each chunk is assigned a monotonically increasing `ChunkId` per lane:

  ```python
  cid = self._lane_next_cid[lane_id]
  inflight_lane[cid] = chunk
  for offset, sample_id in enumerate(chunk):
      yield (sample_id, lane_id, cid, offset)
  ```

- `inflight_chunks_per_lane[lane][cid] = WorkChunk` holds all inflight chunks.

### 1.3 Progress + eviction + replay snapshot (today)

`Pipeline._yield_while_notifying` does:

- For a `SampleBatch`:

  ```python
  lane_id = item.lane_ids[0]
  max_chunk_id = max(item.chunk_ids)
  progress_cursors = [
      record.meta.cursor
      for record in item.records
      if record.meta.chunk_id == max_chunk_id
  ]
  engine.notify(lane_id, max_chunk_id, progress_cursors)
  ```

- For a `SampleRecord`:

  ```python
  lane_id = item.meta.lane_id
  max_chunk_id = item.meta.chunk_id
  progress_cursors = [item.meta.cursor]
  engine.notify(lane_id, max_chunk_id, progress_cursors)
  ```

So `Engine.notify` receives:

- A lane id.
- A **single** watermark chunk id (`max_chunk_id`).
- Cursors **only** from that chunk.

`Engine.notify`:

```python
def notify(self, lane_id: int, max_chunk_id: int, cursors: list[SampleCursor]) -> None:
    inflight_lane = self.inflight_chunks_per_lane[lane_id]

    # 1) Evict older inflight chunks
    for cid in list(inflight_lane.keys()):
        if cid < max_chunk_id:
            inflight_lane.pop(cid, None)

    add_k = len(cursors)

    if lane_id not in self._lane_last_cursor:
        self._lane_last_cursor[lane_id] = None

    # 2) Advance per-lane pointer
    cur = self._lane_progress[lane_id]
    seen_offset = cur.offset if (cur.chunk_id == max_chunk_id) else 0
    self._lane_progress[lane_id] = LanePtr(max_chunk_id, seen_offset + add_k)

    # 3) Track last cursor for replay
    if cursors:
        max_cursor = max(cursors)
        previous = self._lane_last_cursor.get(lane_id)
        if previous is None or max_cursor > previous:
            self._lane_last_cursor[lane_id] = max_cursor
```

Interpretation:

- **Eviction**: as soon as an item with `max_chunk_id = N` appears, chunks `< N` are dropped for that lane.
- **LanePtr**: `(chunk_id, offset)` = “we’ve consumed `offset` outputs from chunk `chunk_id`”.
- **Replay sentinel** (today): `_lane_last_cursor[lane]` is the maximum cursor seen so far; currently treated as a **threshold**.

`state_dict()` persists:

- `inflight[lane][cid] = WorkChunk.state_dict()`
- `progress[lane] = {"chunk_id": ..., "offset": ...}`
- `lane_next_cid[lane]`
- `work_source`, `lane_ws_state`
- `replay_cursors[lane] = last_cursor.as_key()`

`load_state_dict()` restores these and republishes a snapshot via `ReplayConfigService`.

### 1.4 Replay today

Today’s conceptual description:

- `ReplayFilter` gets `replay_cursors[lane]` as a high‑water mark.
- It drops items with `cursor <= saved_cursor` and lets later ones pass.

This only works because tail output per lane is assumed to be **monotone in `SampleCursor`** (no cross-chunk reordering at tail).

---

## 2) Fan-out (1→N) and filtering today

This remains the same conceptually.

- **Filtering**: Operators can drop records; runners keep single-threaded semantics.
- **Fan-out**:

  - Operators call `meta.child(i)` in deterministic order.
  - Children share `(chunk_id, chunk_offset, sample_id)` but extend `lineage` with `i`.

This guarantees:

- Each child has a unique `SampleCursor`.
- Cursor order within a base record matches emission order.

### 2.1 Terminology: base offset, fragment, training record, tombstone

To make later sections easier to follow:

- **Base offset**: a logical unit `(chunk_id, chunk_offset)` in a lane’s `WorkChunk`. This is the “original” sample position in the work stream.
- **Fragment**: a piece of data derived from a base offset. Operators may split a base offset into multiple fragments (e.g., token spans, segments).
- **Training record**: the `SampleRecord` (or `SampleBatch` at the tail) that the training loop sees. It may be:
  - One base offset, or
  - A packed combination of fragments from multiple base offsets.
- **Tombstone (new concept)**: a special record with no “real” payload whose sole purpose is to tell the engine “this base offset is finished; no more fragments will appear for it”. Tombstones exist for correctness of **eviction**, not for training. They are described in detail in §4.1.5.

The rest of the plan uses these terms consistently.

---

## 3) Where the current model breaks: packing and cross-chunk shuffle

Even with full determinism, two kinds of problems appear when we add cross-chunk packing/shuffle:

### 3.1 Eviction is too coarse

Current eviction rule:

```python
evict all inflight chunks with cid < max_chunk_id
```

Assumption:

> Once we see any output involving chunk `N`, no future output from this lane will depend on chunks `< N`.

With cross-chunk buffering/packing/shuffle:

- A stateful op can buffer inputs from chunks `N-2`, `N-1`, `N`, emit a record involving chunk `N`, and only later emit records involving `N-1`.
- If we evict chunk `N-1` at the moment we see chunk `N`, we lose its `WorkChunk`, and a checkpoint taken then can’t reproduce all of the buffered outputs.

We need eviction to be based on **per-base-offset completion**, not just chunk id.

### 3.2 ReplayFilter’s `<=` semantics fail under reordering

Once an operator reorders tail output w.r.t. `SampleCursor`:

- A high-water mark rule “drop while `cursor <= saved_cursor`” can drop records that were never seen before the checkpoint.

Example:

- Lane sees records with cursors `[C2, C1, C3, ...]` (non‑monotone).
- Checkpoint after consuming `C2` → `saved_cursor = C2`.
- On replay, “drop while `cursor <= C2`” drops both `C2` and `C1`, but `C1` was never consumed pre-checkpoint → we skip training on `C1`.

We need a replay rule that doesn’t assume cursor monotonicity.

### 3.3 Packing creates multi-parent outputs

Packing operators:

- Combine fragments from multiple base offsets `(chunk_id, offset)` into a single training record.
- Each packed record may “consume” several base offsets and may “close” several offsets at once.

Eviction now needs to know exactly which base offsets are complete, per chunk; per-record or per-chunk counts are not enough.

---

## 4) Updated extensions to support packing + cross‑chunk shuffle

We now clearly separate:

1. **Eviction correctness** (engine / checkpoint integrity):  
   When is it safe to evict a `WorkChunk`?

2. **Replay deduplication** (ReplayFilter behavior):  
   When resuming from a checkpoint, how do we avoid re-emitting records already consumed by the model?

We’ll add:

- Richer metadata (`FragmentRef` and `contributors`) to know which base offsets a record touches.
- A **tombstone** concept to allow operators to mark offsets complete even when they drop data.
- An equality-based replay rule using one stored cursor per lane.

---

### 4.0 New concept: fragments and tombstones

#### 4.0.1 Fragments

We distinguish:

- **Base sample / base offset**: one index in a `WorkChunk`: `(chunk_id, offset)`.
- **Fragment**: a logical piece of that base sample that flows through the pipeline (possibly after multiple fan‑outs).

A base sample can yield many fragments over the pipeline:

- Early ops: decode, tokenize, etc.
- Later ops: split into windows, pack, etc.

Any tail record may include multiple fragments, possibly from multiple base samples.

#### 4.0.2 Why we need to know “the last fragment”

For eviction, the engine needs to know:

> “Has the pipeline *finished* with base sample `(chunk_id, offset)`?”

We cannot just count records:

- Some fragments may be filtered out.
- Some fragments may be packed together with others.
- Cross-chunk shuffle means fragments can appear out of chunk order.

We want:

- A **single logical marker** saying:
  > “No more fragments that depend on `(chunk_id, offset)` will ever be emitted.”

That marker is `is_last_fragment=True` for some fragment corresponding to that base offset.

#### 4.0.3 Why we need **tombstones**

There is a subtle but important case:

- Suppose an operator is about to emit a fragment that is marked `is_last_fragment=True` for base offset `(cid, off)` — the fragment that should close this base sample.
- Then, the op decides to drop that fragment (e.g., a downstream filter removes it, or a heuristic says “this sample is useless”).

If we simply drop it:

- The engine never sees any fragment with `is_last_fragment=True` for `(cid, off)`.
- That base offset will **never appear “complete”** from the engine’s point of view.
- Therefore, chunk `cid` cannot be evicted, because one of its offsets is still “open”.

Over time, this can cause:

- Chunks to stay in `inflight_chunks_per_lane` forever.
- Checkpoints to retain unnecessary base data.
- Memory footprint creeping up or eviction logic becoming unsound if we try to guess.

**Tombstone** solves this:

> A **tombstone** is a special record whose only purpose is to tell the engine “this base offset is done”, without carrying any training payload.

In practice:

- A tombstone has:
  - `SampleMeta` with a `FragmentRef` for the base offset.
  - `is_last_fragment=True` for that fragment.
  - A payload that is ignored (or empty).
- Tombstones let us say:
  - “(cid, off) is forever complete, even though we dropped all its real fragments.”

We will:

- Treat tombstones as “invisible” to training (the model doesn’t care about them).
- But treat them as “real” for eviction (they mark offsets complete).

---

### 4.1 Data model extensions

#### 4.1.1 `FragmentRef` – per-contributor last-fragment metadata

Add a new type in `constants.py`:

```python
from dataclasses import dataclass
from zephon.core.constants import SampleCursor

@dataclass(frozen=True, slots=True)
class FragmentRef:
    cursor: SampleCursor
    is_last_fragment: bool = True
```

Intuition:

- A `FragmentRef` is a pointer to a *fragment* derived from a base offset `(chunk_id, offset)`; the `cursor` identifies it.
- `is_last_fragment=True` means:  
  **“This fragment (or a tombstone with the same cursor) is the last thing that will ever be emitted for this base offset.”**

This is how the engine will know that an offset is “done”.

#### 4.1.2 Extend `SampleMeta` with contributor/tombstone tags

`SampleMeta` now keeps contributor/tombstone markers inside `tags` to avoid
schema churn while exposing a stable surface:

```python
@dataclass(frozen=True, slots=True)
class SampleMeta:
    sample_id: SampleId
    lane_id: LaneId
    chunk_id: ChunkId
    chunk_offset: ChunkOffset = 0
    lineage: LineagePath = field(default_factory=tuple)
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def cursor(self) -> SampleCursor: ...

    @property
    def contributors(self) -> tuple[FragmentRef, ...]:
        raw = self.tags.get("_contributors")
        if raw is None:
            return ()
        if isinstance(raw, tuple):
            return raw
        return tuple(raw)

    def with_contributors(
        self, value: Iterable[FragmentRef] | None
    ) -> "SampleMeta":
        tags = dict(self.tags)
        if value:
            tags["_contributors"] = tuple(value)
        else:
            tags.pop("_contributors", None)
        return replace(self, tags=tags)

    @property
    def tombstone(self) -> bool:
        return bool(self.tags.get("_tombstone", False))

    def with_tombstone(self, value: bool = True) -> "SampleMeta":
        tags = dict(self.tags)
        if value:
            tags["_tombstone"] = True
        else:
            tags.pop("_tombstone", None)
        return replace(self, tags=tags)

    def contribution_refs(self) -> tuple[FragmentRef, ...]:
        if self.contributors:
            return self.contributors
        return (FragmentRef(self.cursor, True),)

    def contribution_cursors(self) -> tuple[SampleCursor, ...]:
        return tuple(ref.cursor for ref in self.contribution_refs())
```

Backwards compatibility:

- Existing operators that never set contributors continue to behave as if each record is a single fragment with `is_last_fragment=True`.

Packing/shuffle-aware operators:

- Use `with_contributors(...)` to describe which base fragments a record uses, and `with_tombstone()` to mark tombstones.

#### 4.1.3 Helper APIs: splitting, packing, tombstones

We provide helper functions to keep operators simple and correct:

```python
def fragment_meta(parent: SampleMeta, child_idx: int, *, is_last: bool = False) -> SampleMeta:
    """Create metadata for a fragment derived from `parent`.

    child_idx: index in the deterministic emission order from this parent.
    is_last: True if this fragment is the final fragment for the parent’s base offset.
    """
    new_lineage = parent.lineage + (int(child_idx),)
    cursor = SampleCursor(parent.chunk_id, parent.chunk_offset,
                          parent.sample_id, new_lineage)
    ref = FragmentRef(cursor=cursor, is_last_fragment=is_last)
    base = SampleMeta(
        sample_id=parent.sample_id,
        lane_id=parent.lane_id,
        chunk_id=parent.chunk_id,
        chunk_offset=parent.chunk_offset,
        lineage=new_lineage,
        tags=dict(parent.tags),
    )
    return base.with_contributors((ref,))


def pack_meta(
    primary_cursor: SampleCursor,
    contributors: Iterable[FragmentRef],
    *,
    lane_id: int,
    tags: dict[str, Any] | None = None,
) -> SampleMeta:
    """Build metadata for a packed record.

    primary_cursor: identity for replay (record-level cursor).
    contributors: all fragments whose content is included in this record.
    """
    sid = primary_cursor.sample_id
    meta = SampleMeta(
        sample_id=sid,
        lane_id=lane_id,
        chunk_id=primary_cursor.chunk_id,
        chunk_offset=primary_cursor.chunk_offset,
        lineage=primary_cursor.lineage,
        tags=tags or {},
    ).with_contributors(tuple(contributors))
    return meta


def tombstone_meta(ref: FragmentRef, lane_id: int) -> SampleMeta:
    """Emit a 'tombstone' record that marks a base offset as complete
    without carrying any training payload."""
    last_ref = FragmentRef(cursor=ref.cursor, is_last_fragment=True)
    meta = SampleMeta(
        sample_id=ref.cursor.sample_id,
        lane_id=lane_id,
        chunk_id=ref.cursor.chunk_id,
        chunk_offset=ref.cursor.chunk_offset,
        lineage=ref.cursor.lineage,
        tags={"_tombstone": True},
    ).with_contributors((last_ref,))
    return meta
```

#### 4.1.4 Uniqueness of `meta.cursor` for replay

Replay dedup uses `meta.cursor` as the **record identity**, per lane.

We require:

> On each lane, `meta.cursor` must uniquely identify each emitted *training record* (not each fragment).

For non‑packing operators:

- This is already true when fan‑out uses `.child(i)` and each emitted record gets its own lineage.

For packing:

- The operator should choose a deterministic `primary_cursor` (e.g. one fragment’s cursor, plus an extra lineage step if needed) and ensure:
  - It is unique per emitted record on that lane.
  - It is stable across runs.

`contributors` then track which base fragments are included; replay cares only about the record’s own `cursor`.

#### 4.1.5 Tombstones: what they are and why we need them

**Problem:**  
We want to evict a base offset `(chunk_id, offset)` once we know **no more fragments will be emitted from it**. That’s easy when we always emit at least one fragment:

- The operator can mark the last fragment with `is_last_fragment=True`.
- When the engine sees that, it knows the offset is complete.

But consider:

- A filter operator that drops all fragments from a base offset.
- A packer that chooses to never include any fragment for a given base offset (e.g., filtered out due to length).
- A later stage that decides “this data is invalid, don’t use it”.

Without tombstones:

- The engine never sees a fragment with `is_last_fragment=True` for that offset.
- It must assume the offset is still “pending”.
- Result: the containing chunk is never considered fully complete → **we can never safely evict that chunk**. This is a leak and breaks eviction.

**Tombstone solution:**

A **tombstone** is a record with:

- A `FragmentRef` for a base offset with `is_last_fragment=True`.
- No meaningful training payload (e.g. payload could be `None` or minimal and dropped later).
- A `tags["_tombstone"] = True` marker (surfaced via `meta.tombstone`) to help downstream ops ignore it if needed.

Semantically:

> A tombstone says: *“I am not sending any actual training data for this base offset, but please mark it complete so the engine can evict it.”*

**Where tombstones appear:**

- Tombstones are produced **inside the pipeline**, by operators that drop final fragments.
- They can be filtered out before the final training consumer or handled as “no-op” samples.
- They exist primarily for **progress/eviction**, not for modeling.

**Example:**

- Base offset `(10, 5)` is decoded into text.
- A filter decides this text is empty; it doesn’t want to emit any training record for it.
- Instead of silently dropping it, the filter:

  - Creates a `FragmentRef(cursor=<cursor-of-this-base>, is_last_fragment=True)`.
  - Emits a tombstone record with `tombstone_meta(ref, lane_id)`.

- The engine sees `is_last_fragment=True` for `(10,5)` and can mark that offset complete and eventually evict the chunk when all offsets are complete.

---

### 4.2 Eviction & progress accounting (engine correctness)

**Goal:** Do not evict a `WorkChunk` while any of its base offsets `(chunk_id, offset)` may still have fragments that will be emitted later.

We enforce a contract:

> For each base `(chunk_id, offset)` and lane, exactly one fragment (or tombstone) across the pipeline will have `FragmentRef.is_last_fragment=True`. That emission happens after all other fragments for that offset have been emitted or dropped.

Then:

- A base offset `(cid, off)` is **complete** once we have delivered any record whose `contributors` contain a `FragmentRef` where:
  - `ref.cursor.chunk_id == cid`
  - `ref.cursor.chunk_offset == off`
  - `ref.is_last_fragment == True`

Tombstones are the special case where we “complete” an offset without emitting a training record.

#### 4.2.1 Runtime per-offset completion tracking

We keep **runtime-only** structures, per lane:

```python
self._offset_done: dict[LaneId, dict[ChunkId, OffsetBitmap]] = {}
self._offset_done_count: dict[LaneId, dict[ChunkId, int]] = {}
```

Where `OffsetBitmap` is a tiny helper:

```python
class OffsetBitmap:
    def __init__(self, size: int):
        self.size = size
        self._bits = set[int]()  # could be a real bitset instead

    def set(self, offset: int) -> None:
        self._bits.add(offset)

    def is_set(self, offset: int) -> bool:
        return offset in self._bits
```

**Important:**

- This state is not persisted to checkpoints.
- It’s rebuilt after restart by replaying inflight chunks and seeing `is_last_fragment` again.

#### 4.2.2 Updated `Engine.notify` for eviction

We change `notify` to accept a list of contributor refs instead of `(max_chunk_id, cursors)`:

```python
def notify(self, lane_id: int, entries: list[ContributorRef]) -> None:
    inflight_lane = self.inflight_chunks_per_lane[lane_id]
    done = self._offset_done.setdefault(lane_id, {})
    done_count = self._offset_done_count.setdefault(lane_id, {})

    # 1) Update per-offset completion state
    for entry in entries:
        cid = int(entry.cursor.chunk_id)
        off = int(entry.cursor.chunk_offset)

        if cid not in inflight_lane:
            # Chunk already evicted or not relevant
            continue

        if cid not in done:
            chunk = inflight_lane[cid]
            done[cid] = OffsetBitmap(size=len(chunk))
            done_count[cid] = 0

        if entry.is_last_child and not done[cid].is_set(off):
            done[cid].set(off)
            done_count[cid] += 1

    # 2) Evict chunks that are fully complete (in increasing cid order)
    for cid in sorted(list(inflight_lane.keys())):
        chunk = inflight_lane[cid]
        if cid not in done:
            # No last fragments for this chunk yet -> incomplete
            break
        if done_count[cid] == len(chunk):
            # All offsets complete -> safe to evict
            inflight_lane.pop(cid, None)
            done.pop(cid, None)
            done_count.pop(cid, None)
        else:
            # First incomplete chunk; cannot evict beyond this
            break

    # 3) Maintain LanePtr for fairness/diagnostics
    if inflight_lane:
        front_cid = min(inflight_lane.keys())
        front_done = done_count.get(front_cid, 0)
        self._lane_progress[lane_id] = LanePtr(front_cid, front_done)

    # 4) _lane_last_cursor is still updated based on record-level cursors (see replay).
```

Differences vs today:

- Eviction is no longer based on `max_chunk_id`; it is based on **per-offset completion**.
- We do not need to store `_offset_done` in checkpoints; it is runtime-only.

---

### 4.3 Replay correctness (strong replay with one cursor per lane)

Replay is **record-level** and independent of per-offset completion.

We switch from a threshold (`<=`) scheme to an **equality sentinel** scheme:

> For each lane, remember the exact cursor of the last delivered record at checkpoint time. On replay, per lane, drop everything until we see that record again, drop it too, then forward everything after it.

This works even if tail output is not monotone in `SampleCursor`.

#### 4.3.1 Stored replay state (minimal per-lane sentinel)

In `_state_dict_local`:

```python
replay_cursors: dict[int, Any] = {}
for lane, cursor in self._lane_last_cursor.items():
    replay_cursors[int(lane)] = cursor.as_key() if cursor is not None else None
```

This is **one cursor per lane** (or `None` if nothing has been consumed).

In `load_state_dict`:

```python
replay_raw = state.get("replay_cursors", {}) or {}
self._lane_last_cursor = dict.fromkeys(owned)
for lane in owned:
    payload = replay_raw.get(str(lane)) or replay_raw.get(int(lane))
    if payload is None:
        self._lane_last_cursor[lane] = None
    else:
        self._lane_last_cursor[lane] = SampleCursor.from_key(payload)
```

Then `_publish_replay_snapshot()` sends `lane -> SampleCursor|None` to `ReplayConfigService`.

#### 4.3.2 ReplayFilter: flip when we see the sentinel

For each lane `L`:

- `target = snapshot.get(L)` (a `SampleCursor` or `None`).
- `seen_target[L]` is initialized to:
  - `True` if `target is None` (no previous progress; nothing to skip).
  - `False` otherwise.

For each record `R` from lane `L`:

```python
target = snapshot.get(L)
if target is None:
    # No previous progress; everything is new
    yield R
else:
    c = R.meta.cursor
    if not seen_target[L]:
        if c == target:
            # Exactly the last record from previous run
            seen_target[L] = True
            # Drop it as well so resume is exclusive
            continue
        else:
            # Still in already-consumed prefix
            continue
    else:
        # Past the boundary; everything is new
        yield R
```

Requirements:

1. Per lane, `meta.cursor` must uniquely identify each tail record.
2. After `load_state_dict()`, the pipeline is deterministic and the record with `cursor == target` appears again.

With those, we get:

- **Strong replay**: after restart, the model sees exactly the suffix `[R(k+1), R(k+2), ...]` that it would have seen in a crash-free run.

Note: `ReplayFilter` does **not** look at `contributors` or `is_last_fragment`; those are used only for eviction.

---

### 4.4 State persistence summary (minimal checkpoint footprint)

With the new design, `Engine.state_dict()` persists:

- `version`
- `world` (canonical_replicas, num_ranks, mapping, etc.)
- `inflight[lane][cid] = WorkChunk.state_dict()`
- `progress[lane] = LanePtr(chunk_id, offset)` (front chunk + completed offsets count)
- `lane_next_cid[lane]`
- `work_source` (global state)
- `lane_ws_state[lane]`
- `last_round_id`, `checkpoint_reload_count`, `rr_next_idx`
- `replay_cursors[lane]` (sentinel `SampleCursorKey` per lane, or `None`)

We **do not** persist:

- Per-offset bitmaps (`_offset_done`).
- Per-chunk completion counts (`_offset_done_count`).

Those are rebuilt via deterministic replay of inflight chunks.

Checkpoint memory stays basically the same plus one cursor per lane.

---

### 4.5 Operator-facing guidance (updated)

This section clarifies what packing/shuffle operators must do to cooperate with the engine.

#### 4.5.1 Fragment lineage & last-fragment discipline

For each base `(chunk_id, chunk_offset, sample_id)` in a lane:

- The pipeline may emit 0 or more fragments derived from it.
- Exactly **one** fragment (or tombstone) will carry `FragmentRef.is_last_fragment=True`, indicating “no more fragments for this base offset”.

Splitting operators:

- Consume inputs in the order provided per lane.
- For each base record, use `fragment_meta(parent, idx, is_last=...)`:

  - `idx` increases with emission order.
  - Exactly one child has `is_last=True` (typically the last emitted child).

- If all fragments are dropped:
  - Emit a `tombstone_meta(ref)` somewhere downstream to mark the base offset complete.

#### 4.5.2 Packing contract

A packing operator:

- Receives `SampleRecord`s (possibly already fragmented).
- Chooses fragments to combine into a packed record.
- For each packed record:

  - Determines a deterministic `primary_cursor` (record identity).
  - Builds `SampleMeta` with `pack_meta(primary_cursor, contributors, lane_id)` where:

    - `contributors` = all `FragmentRef`s whose content is in this record.
    - Any contributor that closes a base offset has `is_last_fragment=True`.

- Must ensure `primary_cursor` (and thus `meta.cursor`) is **unique per emitted record** for that lane.

#### 4.5.3 Tombstones and filtering

**Why tombstones are necessary (restate):**

- Without tombstones, any base offset whose last fragment is dropped without a `is_last_fragment=True` emission will never be marked complete.
- That blocks chunk eviction forever for that offset’s chunk.

**How tombstones are used:**

- If an operator decides that a base offset’s final fragment should not become part of any training record:

  - It creates a `FragmentRef` for that offset (or reuses an existing one).
  - Emits a tombstone via `tombstone_meta(ref, lane_id)`.

- The engine sees `is_last_fragment=True` and marks the base offset complete.
- Downstream:

  - A filter op can drop tombstones entirely (e.g., if it looks at `tags["_tombstone"]`).
  - Or they can be passed through as no-op payloads (model ignores them).

**Concrete example:**

- A dedup filter finds that the text of base offset `(10,5)` is a duplicate and shouldn’t be trained on.
- Rather than silently dropping everything, it emits a tombstone record:

  - `meta = tombstone_meta(FragmentRef(cursor=(10,5,...), is_last_fragment=True), lane_id)`
  - payload can be `None` or `{}`.

- The engine marks `(10,5)` complete and is free to evict chunk 10 once all its offsets are complete.

#### 4.5.4 Replay awareness

Operators themselves **do not need to know about replay**:

- Replay dedup is handled by ReplayFilter using the equality strategy on `meta.cursor`.
- Operators only need to:

  - Emit deterministic sequences.
  - Respect the fragment/last-fragment/tombstone contract.
  - Ensure `meta.cursor` is unique per emitted record per lane.

---

### 4.6 Constraints and guarantees (updated)

With these changes:

- **Determinism**:

  - For a fixed `WorkSource` state, plan, and environment, the tail stream per lane is deterministic.
  - Each record has a unique `meta.cursor`.

- **Eviction correctness**:

  - A `WorkChunk` is evicted only when **all** its base offsets have had a last fragment or tombstone (`is_last_fragment=True`) observed.
  - No operator can depend on base data from an evicted chunk.

- **Replay correctness (strong)**:

  - For each lane, if we checkpoint after record `Rk`, then on resume:

    - ReplayFilter drops `R0..Rk` again (using equality with `cursor(Rk)`).
    - Emits `R(k+1)..` in the same order as in a crash-free run.

- **Checkpoint memory footprint**:

  - No per-offset replay state is stored; only one `SampleCursorKey` per lane is added.
  - Per-offset bitmaps are runtime-only and small.

---

### 4.7 Examples

#### 4.7.1 Cross-chunk shuffle, no packing

Tail output for lane 0 (conceptual):

```text
Original:  [A, B, C, D, E, ...]
```

where:

- `A` comes from `chunk_id=1`, `B` from `chunk_id=3`, `C` from `chunk_id=2`, etc.
- Cursor order in `(chunk_id, chunk_offset, lineage, sample_id)` may be arbitrary relative to emission order.

Checkpoint after consuming `D`:

- Engine stores `last_cursor[0] = cursor(D)`.

On resume:

- Engine reconstructs inflight chunks and state.
- Deterministically, tail output is again `[A, B, C, D, E, ...]`.

ReplayFilter for lane 0:

- `target = cursor(D)`, `seen_target = False`.
- For `A`, `B`, `C`: `cursor != target`, `seen_target=False` → drop.
- For `D`: `cursor == target`, set `seen_target=True`, drop.
- For `E`, `...`: `seen_target=True` → emit.

Result: the model sees exactly `[E, ...]`, matching the suffix of the crash-free run.

#### 4.7.2 Packing across chunks with last-fragment and tombstones

Lane 0 has two chunks:

- Chunk 10: offsets 0, 1
- Chunk 11: offsets 0, 1

An upstream op splits and packs like:

- `(10,0)` → fragments `f10_0a`, `f10_0b`
- `(10,1)` → fragment `f10_1`
- `(11,1)` → fragment `f11_1`
- `(11,0)` is filtered out completely

Packing:

- `R0` = pack(`f10_0a`, `f10_0b`) with contributors:

  ```python
  [
      FragmentRef(f10_0a_cursor, is_last_fragment=False),
      FragmentRef(f10_0b_cursor, is_last_fragment=True),   # closes (10,0)
  ]
  ```

- `R1` = pack(`f10_1`, `f11_1`) with:

  ```python
  [
      FragmentRef(f10_1_cursor, is_last_fragment=True),    # closes (10,1)
      FragmentRef(f11_1_cursor, is_last_fragment=True),    # closes (11,1)
  ]
  ```

- `(11,0)` is filtered entirely; the filter emits a tombstone:

  ```python
  T = tombstone_meta(FragmentRef(cursor=f11_0_cursor, is_last_fragment=True), lane_id)
  ```

Engine’s view:

- After `R0`, `(10,0)` is complete.
- After `R1`, `(10,1)` and `(11,1)` are complete.
- After `T`, `(11,0)` is complete.
- Chunk 10’s offsets 0 and 1 are complete → chunk 10 can be evicted.
- Chunk 11’s offsets 0 and 1 are complete → chunk 11 can be evicted.

Replay:

- Uses only record-level `meta.cursor` for dedup; tombstones and contributors are used solely for eviction.

---

## 5) Concrete implementation plan

### 5.1 Data model changes (`constants.py`)

1. **Add `FragmentRef`**:

   ```python
   @dataclass(frozen=True, slots=True)
   class FragmentRef:
       cursor: SampleCursor
       is_last_fragment: bool = True
   ```

2. **Extend `SampleMeta`**:

   - Store contributors/tombstone inside `tags` under `_contributors` / `_tombstone`.
   - Provide `with_contributors(...)`, `with_tombstone(...)`, and read-only
     properties `contributors` / `tombstone` that surface those tags.
   - `contribution_refs()` still defaults to `(FragmentRef(self.cursor, True),)` when
     contributors are absent.

3. **Clarify in docs**:

   - `SampleCursor` is the **record-level identity** used by ReplayFilter.
   - It must be unique per emitted record per lane.

### 5.2 Operator helpers

Add a shared utilities module with:

- `fragment_meta(parent: SampleMeta, child_idx: int, *, is_last: bool=False) -> SampleMeta`
- `pack_meta(primary_cursor: SampleCursor, contributors: Iterable[FragmentRef], lane_id: int, tags: dict | None = None) -> SampleMeta`
- `tombstone_meta(ref: FragmentRef, lane_id: int) -> SampleMeta`

Update operators:

- Splitting ops: use `fragment_meta`.
- Packing ops: use `pack_meta` and respect `is_last_fragment` on contributors.
- Filters that drop final fragments: emit `tombstone_meta`.

### 5.3 Pipeline notify path (`Pipeline._yield_while_notifying`)

Change from:

```python
engine.notify(lane_id, max_chunk_id, progress_cursors)
```

to:

1. Build `entries`:

   ```python
   entries: list[ContributorRef] = []

   if isinstance(item, SampleBatch):
       assert len(set(item.lane_ids)) == 1
       lane_id = item.lane_ids[0]
       for rec in item.records:
           for ref in rec.meta.contribution_refs():
               entries.append(ref)

   elif isinstance(item, SampleRecord):
       lane_id = item.meta.lane_id
       for ref in item.meta.contribution_refs():
           entries.append(ref)

   else:
       raise TypeError("Unsupported element type")
   ```

2. Call:

   ```python
   engine.notify(lane_id, entries)
   yield item
   ```

`max_chunk_id` is no longer needed for eviction.

### 5.4 Engine notify + runtime eviction state

Modify `Engine`:

1. Add runtime-only fields:

   ```python
   self._offset_done: dict[LaneId, dict[ChunkId, OffsetBitmap]] = defaultdict(dict)
   self._offset_done_count: dict[LaneId, dict[ChunkId, int]] = defaultdict(dict)
   ```

2. Implement `OffsetBitmap` as above (either set or bitset).

3. Change `notify` to use `entries: list[ProgressEntry]` and:

   - Update `_offset_done` and `_offset_done_count`.
   - Evict chunks when `done_count[cid] == len(chunk)` in cid order.
   - Update `_lane_progress` accordingly.

4. Remove any eviction based on `max_chunk_id`.

### 5.5 Replay state & ReplayFilter

1. **Engine**:

   - Keep `_lane_last_cursor[lane]` as “last delivered record cursor”.
   - In `state_dict()` include `replay_cursors[lane]` using `cursor.as_key()`.
   - In `load_state_dict()`, reconstruct `_lane_last_cursor` and call `_publish_replay_snapshot()`.

2. **ReplayFilter**:

   - Snapshot: `lane -> SampleCursor or None`.
   - Maintain `seen_target[lane]`.
   - Drop until `record.meta.cursor == target`, then drop that record too, then pass everything.

3. **Placement**:

   - ReplayFilter can remain near the tail (before or after batching), even after packers/shufflers, because it only uses record-level identity.

### 5.6 Tests

Add tests to guide and validate implementation:

1. **Data model tests**:

   - `fragment_meta` creates distinct cursors and `contributors` for children.
   - `pack_meta` preserves contributors and record cursor.
   - `tombstone_meta` sets `is_last_fragment=True` and applies `tags["_tombstone"]`.

2. **Eviction tests**:

   - Single chunk, multiple offsets, fragments arriving in arbitrary order:
     - Verify chunks aren’t evicted prematurely.
   - Multiple chunks:
     - Interleave last fragments for chunks 10 and 11.
     - Verify chunk 10 is only evicted when all its offsets complete, regardless of chunk 11 emissions.

3. **Cross-chunk shuffle tests**:

   - Op that intentionally emits records from chunk 2 before chunk 1.
   - Old eviction logic (if still present) should fail (xfail).
   - New per-offset eviction should succeed.

4. **Packing tests**:

   - Pack fragments from multiple chunks.
   - Ensure multiple offsets can be completed in one packed record and that chunk eviction still works.

5. **Tombstone tests**:

   - Filter that drops all fragments for some offsets but emits tombstones.
   - Verify those offsets are marked complete and chunks can be evicted.

6. **Replay equality tests**:

   - Tail output with non-monotonic cursor order.
   - Checkpoint after record `Rk`.
   - On replay, ensure suffix after replay is exactly `[R(k+1)..]` per lane.

7. **Multi-lane tests**:

   - Multiple lanes with independent last cursors.
   - Checkpoint mid-run.
   - On replay, check per-lane suffix and overall multiset of records match the crash-free run (up to RR interleaving).
