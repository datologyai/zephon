"""Write msgpack-encoded records into ArrayRecord shards."""

import os

import msgpack
from array_record.python.array_record_module import ArrayRecordWriter

records = [{"text": f"sample {i}", "label": i % 2} for i in range(1000)]
os.makedirs("data/array_record_demo", exist_ok=True)

# One file per shard. group_size:1 lets each record be decoded independently;
# larger groups can improve compression.
for shard, start in enumerate(range(0, len(records), 250)):
    path = f"data/array_record_demo/shard{shard}.array_record"
    writer = ArrayRecordWriter(path, "group_size:1")
    for record in records[start : start + 250]:
        writer.write(msgpack.packb(record))
    writer.close()
