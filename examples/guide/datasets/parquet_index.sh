# Build an index.json so Zephon can plan without reading every footer.

python -m zephon.build_index parquet s3://my-bucket/corpora/wikipedia
