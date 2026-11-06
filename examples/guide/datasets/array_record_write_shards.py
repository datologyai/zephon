"""Write MessagePack-encoded pretraining documents into ArrayRecord shards."""

import os

import msgpack
from array_record.python.array_record_module import ArrayRecordWriter

records = [
    {"text": "The ocean covers most of Earth's surface."},
    {"text": "Plants use sunlight to turn water and carbon dioxide into sugars."},
    {"text": "Written language allows knowledge to be shared across generations."},
]
os.makedirs("data/array_record_demo", exist_ok=True)

# One file per shard. group_size:1 lets each record be decoded independently;
# larger groups can improve compression.
for shard, start in enumerate(range(0, len(records), 2)):
    path = f"data/array_record_demo/shard{shard}.array_record"
    writer = ArrayRecordWriter(path, "group_size:1")
    for record in records[start : start + 2]:
        writer.write(msgpack.packb(record))
    writer.close()
