# Data storage

- `samples/`: tiny synthetic fixtures tracked in Git; never measured scientific data.
- `snapshots/`: small public source indexes with provenance, when available.
- `raw/`: downloaded original archives; ignored by Git.
- `processed/`: harmonized NPZ files and manifests; ignored by Git.

Dataset catalog: `waste_lab/catalog.json`. All candidates have a source, license caveat and intended use. A catalog entry does not mean the dataset was downloaded or integrated.

Do not commit API keys, paid imagery or bulk datasets. The downloader saves SHA256 provenance and enforces a size cap. Zenodo downloads also validate the provider checksum. Archive extraction is intentionally manual; inspect archive paths before extracting.
