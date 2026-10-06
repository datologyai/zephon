"""Define and inspect a Vortex Dataset."""

from zephon.debug import DatasetInspector
from zephon.io import Dataset

# Vortex files are self-describing, so the index is an optimization, not a
# requirement. Compression is handled inside the format: no extra local copy.
dataset = Dataset.from_path("vortex_demo", "data/vortex_demo", fmt="vortex")

with DatasetInspector(dataset) as inspector:
    print(inspector.read(shard_id=0, sample_index=0))
