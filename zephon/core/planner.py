# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage planner that converts logical graphs into executable plans."""

from zephon.core.graph import Graph, Node, Plan, Stage


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

        for node in graph.nodes:
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

        indexable = all(
            all(nd.op.traits().indexable for nd in stage.nodes) for stage in stages
        )
        explain_lines = []
        for idx, stage in enumerate(stages):
            ops = [f"{nd.name}@p{nd.parallelism}" for nd in stage.nodes]
            explain_lines.append(
                f"Stage[{idx}] place={stage.placement} break='{stage.break_reason}' ops={ops}"
            )
        return Plan(
            stages=stages, explain="\n".join(explain_lines), indexable=indexable
        )
