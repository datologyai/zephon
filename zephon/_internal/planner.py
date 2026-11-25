# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage planner that converts logical graphs into executable plans."""

import dataclasses

from zephon._internal.branching import stateless_violation
from zephon._internal.graph import Graph, Merge, Node, Plan, Split, Stage, Vertex
from zephon._internal.ops.batch import Batch
from zephon._internal.ops.map_transform import MapBatchTransform, MapTransform
from zephon._internal.ops.replay_filter import ReplayFilter


class Planner:
    """Compile a `Graph` of ops into an execution `Plan` for the data loader.

    Internal machinery. Do not drive this directly — build and run pipelines via
    ``Pipeline``.

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
        branch_ops = self._branch_ops(graph)
        nodes = self._with_replay_filters(graph)
        self._validate_map_batch_ordering(nodes)
        self._validate_branch_ops(graph.merges, branch_ops)

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
        self._place_branch_points(stages, graph.merges, branch_ops)

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
            ops = self._explain_stage(stage, branch_ops)
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

    @staticmethod
    def _explain_stage(
        stage: Stage, branch_ops: dict[int, dict[str, list[Node]]]
    ) -> list[str]:
        """Describe a stage's ops for ``Plan.explain``.

        Branch ops carry their branch name. Each split shows up before its
        first branch op and each merge after its last one.
        """
        position = {id(node): index for index, node in enumerate(stage.nodes)}
        branch_of: dict[int, str] = {}
        opens: dict[int, Split] = {}
        closes: dict[int, Merge] = {}
        for merge in stage.merges:
            ops: list[Node] = []
            for name, chain in branch_ops[id(merge)].items():
                for node in chain:
                    branch_of[id(node)] = name
                    ops.append(node)
            ops.sort(key=lambda node: position[id(node)])
            opens[id(ops[0])] = merge.split
            closes[id(ops[-1])] = merge

        entries: list[str] = []
        for node in stage.nodes:
            split = opens.get(id(node))
            if split is not None:
                entries.append(f"{split.name}({', '.join(split.router.branches)})")
            text = f"{node.name}@p{node.parallelism}"
            branch = branch_of.get(id(node))
            entries.append(text if branch is None else f"[{branch}] {text}")
            merge = closes.get(id(node))
            if merge is not None:
                entries.append(merge.name)
        return entries

    def _branch_ops(self, graph: Graph) -> dict[int, dict[str, list[Node]]]:
        """Check every split and merge pair and return each merge's branch ops.

        Each split needs exactly one merge with the same branches, must come
        before any Batch (a split routes records, not batches), and feeds only
        the first op of each branch. An op that reads a split or merge reads
        nothing else, and only the next op in its branch or the merge reads a
        branch op.

        Returns:
            For each merge (keyed by ``id``), branch name to that branch's ops,
            from the op that reads the split to the merge's tail.
        """
        position = {id(node): index for index, node in enumerate(graph.nodes)}
        first_batch = next(
            (
                index
                for index, node in enumerate(graph.nodes)
                if isinstance(node.op, Batch)
            ),
            None,
        )
        merges_of: dict[int, list[Merge]] = {id(split): [] for split in graph.splits}
        for merge in graph.merges:
            if id(merge.split) not in merges_of:
                raise ValueError(
                    f"merge {merge.name!r} rejoins split {merge.split.name!r}, "
                    "which is not in the graph"
                )
            merges_of[id(merge.split)].append(merge)
        for split in graph.splits:
            count = len(merges_of[id(split)])
            if count != 1:
                raise ValueError(
                    f"split {split.name!r} has {count} merges; it needs exactly one"
                )
            if id(split.input) not in position:
                raise ValueError(
                    f"split {split.name!r} reads {split.input.name!r}, which is "
                    "not in the graph"
                )
            if first_batch is not None and position[id(split.input)] >= first_batch:
                raise ValueError(
                    "branch() cannot come after batch(): a branch routes records, "
                    "but after batch() the stream carries batches. Call branch() "
                    "before batch()."
                )

        readers: dict[int, list[Node]] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if isinstance(inp, (Split, Merge)):
                    if len(node.inputs) != 1:
                        raise ValueError(
                            f"op {node.name!r} reads {inp.name!r} and other "
                            "inputs; an op after a split or merge reads only it"
                        )
                    readers.setdefault(id(inp), []).append(node)

        result: dict[int, dict[str, list[Node]]] = {}
        next_op: dict[int, Node | None] = {}
        for merge in graph.merges:
            split = merge.split
            if tuple(merge.branches) != split.router.branches:
                raise ValueError(
                    f"merge {merge.name!r} rejoins branches {list(merge.branches)}, "
                    f"but split {split.name!r} routes to {list(split.router.branches)}"
                )
            chains = {name: self._walk_branch(merge, name) for name in merge.branches}
            firsts = [chain[0] for chain in chains.values()]
            split_readers = readers.get(id(split), [])
            if len(split_readers) != len(firsts) or {id(n) for n in split_readers} != {
                id(n) for n in firsts
            }:
                raise ValueError(
                    f"split {split.name!r} must feed the first op of each of its "
                    "branches and nothing else"
                )
            if len(readers.get(id(merge), [])) > 1:
                raise ValueError(f"merge {merge.name!r} feeds more than one op")
            for chain in chains.values():
                for node, after in zip(chain, [*chain[1:], None]):
                    next_op[id(node)] = after
            result[id(merge)] = chains

        for node in graph.nodes:
            for inp in node.inputs:
                if id(inp) in next_op and next_op[id(inp)] is not node:
                    raise ValueError(
                        f"op {node.name!r} reads branch op {inp.name!r} directly; "
                        "ops after the branches read their merge"
                    )
        return result

    @staticmethod
    def _walk_branch(merge: Merge, name: str) -> list[Node]:
        """Return branch ``name``'s ops, from the op that reads the split to its tail."""
        chain = [merge.branches[name]]
        while len(chain[-1].inputs) == 1:
            (inp,) = chain[-1].inputs
            if inp is merge.split:
                return chain[::-1]
            if not isinstance(inp, Node):
                break
            chain.append(inp)
        raise ValueError(
            f"branch {name!r} of merge {merge.name!r} does not lead back to "
            f"split {merge.split.name!r}"
        )

    @staticmethod
    def _validate_branch_ops(
        merges: list[Merge], branch_ops: dict[int, dict[str, list[Node]]]
    ) -> None:
        """Reject branch ops that are not stateless and record-local.

        The runner flushes a branch op after every slice of the split's
        output, so its output must not depend on how its input is cut into
        batches.
        """
        for merge in merges:
            for branch, chain in branch_ops[id(merge)].items():
                for node in chain:
                    reason = stateless_violation(node)
                    if reason is not None:
                        raise ValueError(
                            f"op {node.name!r} in branch {branch!r} cannot run in "
                            f"a branch: {reason}. Branches take only stateless, "
                            "record-local ops; add this op after the branch() call."
                        )

    @staticmethod
    def _place_branch_points(
        stages: list[Stage],
        merges: list[Merge],
        branch_ops: dict[int, dict[str, list[Node]]],
    ) -> None:
        """Put each split and merge in the stage that runs their branch ops."""
        stage_of = {id(node): stage for stage in stages for node in stage.nodes}
        for merge in merges:
            chains = branch_ops[id(merge)].values()
            homes = [stage_of[id(node)] for chain in chains for node in chain]
            if any(home is not homes[0] for home in homes):
                raise ValueError(
                    f"the branches after {merge.split.input.name!r} landed in "
                    "different stages; give every branch op the same placement"
                )
            homes[0].splits.append(merge.split)
            homes[0].merges.append(merge)

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
        """Return the plan's node order with a ReplayFilter before each Batch or at the tail.

        The user's graph is left untouched. The Pipeline caches plans and
        drops them when the graph changes, so a ReplayFilter written into the graph
        would survive that reset and end up in the middle of the chain once more
        ops are added. On restore, two filters on one lane drop every record after the
        replay cursor.
        """
        # Only this method creates ReplayFilter nodes, and it never adds them to the graph.
        assert not any(isinstance(node.op, ReplayFilter) for node in graph.nodes), (
            "ReplayFilter nodes are added by the planner and must not appear in the graph"
        )
        nodes: list[Node] = []
        has_batch = False
        # Ops that read a copied node read the copy instead, so every edge in
        # the plan points at a vertex of the plan.
        replaced: dict[int, Node] = {}
        for original in graph.nodes:
            node = original
            inputs: list[Vertex] = [replaced.get(id(inp), inp) for inp in node.inputs]
            if isinstance(node.op, Batch):
                has_batch = True
                # ReplayFilter must be single-threaded per stage to maintain per-lane equality replay semantics.
                dop = 1
                filter_node = Node(
                    name=f"{node.name}_replay_filter",
                    op=ReplayFilter(),
                    inputs=inputs,
                    placement=node.placement,
                    parallelism=dop,
                )
                # TODO: Explore co-locating ReplayFilter within Batch once we have explicit pre-batch hooks.
                # Rewire a copy so the user's Batch node keeps its original inputs.
                node = dataclasses.replace(node, inputs=[filter_node])
                nodes.append(filter_node)
            elif any(new is not old for new, old in zip(inputs, node.inputs)):
                node = dataclasses.replace(node, inputs=inputs)
            if node is not original:
                replaced[id(original)] = node
            nodes.append(node)

        if not has_batch and nodes:
            tail = self._tail(graph)
            # Keep replay filtering single-threaded at the tail.
            dop = 1
            filter_node = Node(
                name=f"{tail.name}_replay_filter",
                op=ReplayFilter(),
                inputs=[tail],
                placement=nodes[-1].placement,
                parallelism=dop,
            )
            nodes.append(filter_node)

        return nodes

    @staticmethod
    def _tail(graph: Graph) -> Node | Merge:
        """Return the vertex the pipeline ends with.

        That is the last op, unless the pipeline ends right after
        ``branch()``: then it is the merge nothing reads.
        """
        read = {id(inp) for node in graph.nodes for inp in node.inputs}
        unread = [merge for merge in graph.merges if id(merge) not in read]
        return unread[-1] if unread else graph.nodes[-1]
