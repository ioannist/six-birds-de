# Data directory

- raw/ holds fetched or copied source datasets (verbatim).
- processed/ holds derived datasets created by our pipelines.
- Use `python scripts/fetch_data.py --dataset <name>` to populate raw/.
- Checksums are verified and provenance is recorded for each fetch.
- data/raw/** and data/processed/** are gitignored.
- data/registry.yaml is tracked and defines dataset sources.
