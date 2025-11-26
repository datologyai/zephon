# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage planner that converts logical graphs into executable plans."""

from zephon.core.graph import Graph, Node, Plan, Stage
from zephon.ops.batch import Batch
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

        indexable = all(
            all(nd.op.traits().indexable for nd in stage.nodes) for stage in stages
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
            batch_size_hint=batch_size_hint,
        )

    def _split_batch_stages(self, stages: list[Stage]) -> list[Stage]:
        expanded: list[Stage] = []
        for stage in stages:
            expanded.extend(self._split_stage_for_batch(stage))
        return expanded

    def _split_stage_for_batch(self, stage: Stage) -> list[Stage]:
        nodes = stage.nodes
        if not any(isinstance(nd.op, Batch) for nd in nodes):
            return [stage]

        segments: list[tuple[list[Node], bool]] = []
        idx = 0
        total = len(nodes)
        while idx < total:
            batch_idx = next(
                (i for i in range(idx, total) if isinstance(nodes[i].op, Batch)),
                None,
            )
            if batch_idx is None:
                if idx < total:
                    segments.append((nodes[idx:], False))
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

            if inline_start > idx:
                segments.append((nodes[idx:inline_start], False))

            segments.append((nodes[inline_start : batch_idx + 1], True))
            idx = batch_idx + 1

        result: list[Stage] = []
        for seg_idx, (seg_nodes, inline) in enumerate(segments):
            if not seg_nodes:
                continue
            name = stage.name if seg_idx == 0 else f"{stage.name}#{seg_idx}"
            break_reason = stage.break_reason if seg_idx == 0 else "batch-inline"
            runner_hint = "inline" if inline else stage.runner_hint
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
                    dop = node.parallelism or 1
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
                dop = tail.parallelism or 1
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
