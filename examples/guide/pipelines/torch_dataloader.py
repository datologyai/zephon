"""Hand a Pipeline to a torch DataLoader that insists on a Dataset."""

from torch.utils.data import DataLoader

from zephon import Pipeline

pipeline = Pipeline(work_source).batch(microbatch_size=32)

# The Pipeline has already batched and already runs its own workers, so the
# DataLoader must do neither.
loader = DataLoader(pipeline.to_torch_dataset(), batch_size=None, num_workers=0)
