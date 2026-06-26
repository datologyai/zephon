# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage planner that converts logical graphs into executable plans."""

from zephon.core.graph import Graph, Node, Plan, Stage
from zephon.ops.batch import Batch
from zephon.ops.map_transform import MapBatchTransform, MapTransform
from zephon.ops.replay_filter import ReplayFilter


class Planner:
    """Compile a `Graph` of ops into an execution `Plan` for the data loader.

    The pipeline API builds a `Graph` ordering the logical `Node`s that the data
    loader should execute. The `Planner` groups those nodes into pipeline
    `Stage`s so the `Engine` can attach concrete runners (threads, processes,
    remote executors, etc.) and drive the runtime. This keeps graph construction
    decoupled from execution-specific concerns.
    """

    def make_plan(self, graph: Graph) -> Plan:
        """Fuse graph nodes into stages and annotate the resulting `Plan`.

        We walk the graph once, collecting contiguous nodes into the current
        stage until we hit a barrier. The only barrier today is a placement
        change (for example an op that must run "remote" after "auto" work),
        but the structure leaves room for future traits to demand breaks as
        well. When a barrier fires we flush the collected nodes into a `Stage`,
        record the reason ("placement-hint", "barrier", ...), and reset the
        staging buffers. The stage inherits the last explicit placement seen
        inside the group; otherwise it remains "auto". After visiting every
        node we flush the tail to produce the final stage.

        After staging we derive two bits of runtime metadata: a readable
        `explain` string describing each stage and an `indexable` flag that
        stays `True` only when every op advertises `traits().indexable`. The
        `Engine` consumes these annotations to pick stage runners and expose
        diagnostics for the data-loading plan.
        """
        nodes = self._with_replay_filters(graph)
        self._validate_map_batch_ordering(nodes)

        stages: list[Stage] = []
        current_nodes: list[Node] = []
        current_io_bound = False
        current_placement = "auto"
        break_reason = "start"

        def flush(reason: str) -> None:
            nonlocal current_nodes, current_io_bound, current_placement, break_reason
            if current_nodes:
                stage_name = f"{current_nodes[0].name}+"
                stages.append(
                    Stage(
                        name=stage_name,
                        nodes=list(current_nodes),
                        placement=current_placement,
                        break_reason=break_reason,
                    )
                )
            current_nodes = []
            current_io_bound = False
            current_placement = "auto"
            break_reason = reason

        for node in nodes:
            # traits = node.op.traits()
            # Right now, we fuse everything, unless we want to have a new placement (e.g. remote -> local)
            barrier = False
            placement_change = (
                current_nodes
                and current_placement != node.placement
                and node.placement != "auto"
            )
            if barrier or placement_change:
                flush("barrier" if barrier else "placement-hint")
            current_nodes.append(node)
            if node.placement != "auto":
                current_placement = node.placement
        flush("end")

        stages = self._split_batch_stages(stages)

        indexable = True
        preserves_cursor_order = True
        for stage in stages:
            for nd in stage.nodes:
                traits = nd.op.traits()
                indexable = indexable and traits.indexable
                preserves_cursor_order = (
                    preserves_cursor_order and traits.preserves_cursor_order
                )

        batch_size_hint = None
        for stage in stages:
            for nd in stage.nodes:
                if isinstance(nd.op, Batch):
                    batch_size_hint = nd.op.microbatch_size

        explain_lines = []
        for idx, stage in enumerate(stages):
            ops = [f"{nd.name}@p{nd.parallelism}" for nd in stage.nodes]
            explain_lines.append(
                f"Stage[{idx}] place={stage.placement} break='{stage.break_reason}' ops={ops}"
            )
        return Plan(
            stages=stages,
            explain="\n".join(explain_lines),
            indexable=indexable,
            preserves_cursor_order=preserves_cursor_order,
            batch_size_hint=batch_size_hint,
        )

    def _validate_map_batch_ordering(self, nodes: list[Node]) -> None:
        """Ensure map operators appear on the correct side of Batch."""
        seen_batch = False
        for node in nodes:
            if isinstance(node.op, Batch):
                seen_batch = True
            elif isinstance(node.op, MapTransform) and seen_batch:
                raise ValueError(
                    "map_transform() cannot be used after batch(); "
                    "use map_batch() instead."
                )
            elif isinstance(node.op, MapBatchTransform) and not seen_batch:
                raise ValueError("map_batch() requires a preceding batch() operator.")

    def _split_batch_stages(self, stages: list[Stage]) -> list[Stage]:
        expanded: list[Stage] = []
        for stage in stages:
            expanded.extend(self._split_stage_for_batch(stage))
        return expanded

    def _split_stage_for_batch(self, stage: Stage) -> list[Stage]:
        """Split a stage around Batch operators.

        When a Batch op has post-batch operators after it (before the next
        Batch or end of stage), they are merged into the same segment and
        the segment uses the default runner (threads) instead of inline.
        When Batch is the terminal op with nothing after it, the segment
        is kept inline for lower overhead.
        """
        nodes = stage.nodes
        if not any(isinstance(nd.op, Batch) for nd in nodes):
            return [stage]

        segments: list[tuple[list[Node], str | None]] = []
        idx = 0
        total = len(nodes)
        while idx < total:
            batch_idx = next(
                (i for i in range(idx, total) if isinstance(nodes[i].op, Batch)),
                None,
            )
            if batch_idx is None:
                if idx < total:
                    segments.append((nodes[idx:], None))
                break

            inline_start = batch_idx
            if batch_idx > 0:
                prev = nodes[batch_idx - 1]
                current = nodes[batch_idx]
                if (
                    isinstance(prev.op, ReplayFilter)
                    and prev.name == f"{current.name}_replay_filter"
                ):
                    inline_start = batch_idx - 1

            # Pre-batch segment
            if inline_start > idx:
                segments.append((nodes[idx:inline_start], None))

            # Find where this batch's segment ends: next Batch or end of stage.
            next_batch_idx = next(
                (
                    i
                    for i in range(batch_idx + 1, total)
                    if isinstance(nodes[i].op, Batch)
                ),
                None,
            )
            if next_batch_idx is not None:
                # Stop before the next Batch's ReplayFilter if present.
                segment_end = next_batch_idx
                if next_batch_idx > 0:
                    prev_next = nodes[next_batch_idx - 1]
                    next_node = nodes[next_batch_idx]
                    if (
                        isinstance(prev_next.op, ReplayFilter)
                        and prev_next.name == f"{next_node.name}_replay_filter"
                    ):
                        segment_end = next_batch_idx - 1
            else:
                segment_end = total

            has_post_batch_ops = segment_end > batch_idx + 1
            if has_post_batch_ops:
                # Post-batch ops (e.g. tensor construction) must use threads
                # to avoid expensive IPC serialization of tensors across processes.
                segments.append((nodes[inline_start:segment_end], "threads"))
            else:
                segments.append((nodes[inline_start : batch_idx + 1], "inline"))
            idx = segment_end

        result: list[Stage] = []
        for seg_idx, (seg_nodes, hint) in enumerate(segments):
            if not seg_nodes:
                continue
            name = stage.name if seg_idx == 0 else f"{stage.name}#{seg_idx}"
            break_reason = stage.break_reason if seg_idx == 0 else "batch-inline"
            runner_hint = hint if hint is not None else stage.runner_hint
            result.append(
                Stage(
                    name=name,
                    nodes=seg_nodes,
                    placement=stage.placement,
                    break_reason=break_reason,
                    runner_hint=runner_hint,
                )
            )

        return result if result else [stage]

    def _with_replay_filters(self, graph: Graph) -> list[Node]:
        nodes: list[Node] = []
        has_batch = False
        for node in graph.nodes:
            if isinstance(node.op, Batch):
                has_batch = True
                has_filter = any(
                    isinstance(inp.op, ReplayFilter) for inp in node.inputs
                )
                if not has_filter:
                    # ReplayFilter must be single-threaded per stage to maintain per-lane equality replay semantics.
                    dop = 1
                    filter_node = Node(
                        name=f"{node.name}_replay_filter",
                        op=ReplayFilter(),
                        inputs=list(node.inputs),
                        placement=node.placement,
                        parallelism=dop,
                    )
                    # TODO: Explore co-locating ReplayFilter within Batch once we have explicit pre-batch hooks.
                    node.inputs = [filter_node]
                    nodes.append(filter_node)
                nodes.append(node)
            else:
                nodes.append(node)

        if not has_batch and nodes:
            tail = nodes[-1]
            if not isinstance(tail.op, ReplayFilter):
                # Keep replay filtering single-threaded at the tail.
                dop = 1
                filter_node = Node(
                    name=f"{tail.name}_replay_filter",
                    op=ReplayFilter(),
                    inputs=[tail],
                    placement=tail.placement,
                    parallelism=dop,
                )
                nodes.append(filter_node)

        graph.nodes = nodes
        return nodes
