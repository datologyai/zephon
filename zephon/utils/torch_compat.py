"""Helper module to find out whether we are in torch or torchdata."""

import inspect

_TORCHDATA_HINTS = (
    "torchdata.stateful_dataloader",  # e.g. 'torchdata.stateful_dataloader.worker'
    "torchdata._stateful_dataloader",  # defensive (alt/private layouts)
)
_TORCH_VANILLA_HINT = "torch.utils.data"  # e.g. 'torch.utils.data._utils.worker'


def detect_loader_kind() -> str:
    """Decides which dataloader we are using.

    Options:
      - 'torchdata'  (StatefulDataLoader worker)
      - 'vanilla'    (torch.utils.data.DataLoader worker or main-thread)
      - 'unknown'
    Heuristic: scan the whole call stack; torchdata wins if present anywhere.
    """
    saw_torchdata = False
    saw_vanilla = False

    try:
        for fi in inspect.stack():
            fn = (fi.filename or "").replace("\\", "/")
            fr = getattr(fi, "frame", None)
            mod = fr.f_globals.get("__name__", "") if fr else ""
            pkg = fr.f_globals.get("__package__", "") if fr else ""

            # torchdata evidence
            if (
                any(h in mod for h in _TORCHDATA_HINTS)
                or any(h in pkg for h in _TORCHDATA_HINTS)
                or any(f"/{h.replace('.', '/')}/" in fn for h in _TORCHDATA_HINTS)
            ):
                saw_torchdata = True

            # vanilla torch evidence
            if (
                _TORCH_VANILLA_HINT in mod
                or _TORCH_VANILLA_HINT in pkg
                or "/torch/utils/data/" in fn
            ):
                saw_vanilla = True
    except Exception:
        # fall back below
        pass

    if saw_torchdata:
        return "torchdata"
    if saw_vanilla:
        return "vanilla"
    return "unknown"
