"""Write a small litData dataset, index.json included."""

from litdata import optimize


def samples(count: int) -> list[dict[str, object]]:
    """Return the records to serialize, one dict per sample."""
    return [{"text": f"sample {i}", "label": i % 2} for i in range(count)]


def serialize(record: dict[str, object]) -> dict[str, object]:
    """Write each record through unchanged; litdata pickles this to its workers."""
    return record


optimize(
    fn=serialize,
    inputs=samples(1000),
    output_dir="data/litdata_demo",
    chunk_bytes="64MB",
)
