# pyseis-io

A standalone I/O library for seismic data with a validated internal format and support for multiple industry-standard formats.

## Overview

`pyseis-io` is a foundational library extracted from PySeis that provides:

- **Validated Internal Format**: Zarr + Parquet + YAML/JSON with schema-driven validation
- **Multiple Format Support**: Read/write SEG-Y, SEG-D, SU, and export to GeoPackage
- **Lazy Loading**: Efficient handling of large datasets with Dask integration
- **Strict Validation**: Schema integrity, data consistency, and provenance tracking

## Key Features

- **Zarr-based trace storage**: Chunked, compressed, lazy I/O for trace data
- **Parquet headers**: Columnar, efficient slicing for trace/source/receiver metadata
- **Schema validation**: All metadata components validated against YAML schemas
- **Provenance tracking**: Automatic history of all dataset operations
- **Lazy operations**: Data stays on disk until explicitly computed
- **Unified API**: Consistent `SeismicData` interface across all operations

## Installation

```bash
pip install -e .
```

## Quick Start

### Creating a Dataset

```python
import numpy as np
import pandas as pd
from pyseis_io.core.writer import InternalFormatWriter

# Create synthetic data
data = np.random.randn(100, 2000).astype(np.float32)
headers = pd.DataFrame({
    "trace_id": np.arange(100, dtype=np.int32),
    "source_id": ["0"] * 100,
    "receiver_id": ["0"] * 100,
    "cdp_id": ["0"] * 100,
    "trace_sequence_number": np.arange(100, dtype=np.int32),
    "offset": np.linspace(10, 1000, 100, dtype=np.float32),
    "mute_start": np.zeros(100, dtype=np.float32),
    "mute_end": np.zeros(100, dtype=np.float32),
    "total_static": np.zeros(100, dtype=np.float32),
    "trace_identification_code": np.ones(100, dtype=np.int32),
    "correlated": np.zeros(100, dtype=bool),
    "trace_weighting_factor": np.ones(100, dtype=np.float32)
})

# Write dataset
writer = InternalFormatWriter("my_dataset", overwrite=True)
writer.write_traces(data)
writer.write_headers(trace_headers=headers)
writer.write_metadata({"sample_rate": 0.002})
```

### Reading a Dataset

```python
from pyseis_io.core.dataset import SeismicData

# Open dataset
sd = SeismicData.open("my_dataset")

# Access properties
print(f"Traces: {sd.n_traces}, Samples: {sd.n_samples}")
print(f"Sample rate: {sd.sample_rate}")

# Access headers (lazy)
headers_df = sd.headers  # Pandas DataFrame

# Access trace data (lazy Dask array)
traces = sd.data
```

### Slicing and Computing

```python
# Slice dataset (lazy - returns new SeismicData view)
subset = sd[10:20]      # Traces 10-19
decimated = sd[::2]     # Every other trace
single = sd[0]          # Single trace (1D array)

# Materialize data into memory
traces_np, headers_df = sd.compute()
```

### Saving Derived Datasets

```python
# Save a subset
sd[0:50].save("subset_dataset", overwrite=True)
```

## Dataset Structure

Every dataset follows a strict on-disk layout:

```
<dataset_root>/
    traces.zarr/                 # Zarr group containing trace data
        data/                    # Chunked, compressed trace array
    
    trace.parquet                # Trace headers (required)
    source.parquet               # Source metadata (optional)
    receiver.parquet             # Receiver metadata (optional)
    signature.parquet            # Signature data (optional)
    
    metadata/
        metadata.json            # Global metadata (required)
        provenance.yaml          # Operation history
        schema_manifest.yaml     # Schema checksums and versions
        properties.yaml          # Optional properties
        survey.yaml              # Optional survey info
        instrument.yaml          # Optional instrument info
        job.yaml                 # Optional job info
    
    schema/
        <component>/
            vX.Y.yaml            # Installed schema definitions
```

## Core API

### SeismicData

Primary in-memory representation of a dataset:

```python
# Properties
sd.n_traces       # Number of traces
sd.n_samples      # Trace length
sd.headers        # Pandas DataFrame (Arrow-backed)
sd.data           # Dask array
sd.sample_rate    # Sample rate in seconds
sd.file_path      # Dataset location

# Methods
sd.open(path)           # Class method to load dataset
sd.save(path)           # Save to new location
sd.compute()            # Materialize traces and headers
sd.close()              # Release resources
sd[slice]               # Slice dataset (returns new SeismicData)
```

### InternalFormatWriter

Write datasets in the internal format:

```python
writer = InternalFormatWriter(root, overwrite=True)

# Write trace data (NumPy or Dask arrays)
writer.write_traces(data, chunks=None, compressor='blosc', compression_level=5)

# Write headers with schema validation
writer.write_headers(
    trace_headers=df,
    source_headers=None,
    receiver_headers=None
)

# Write metadata
writer.write_metadata({"sample_rate": 0.002})

# Write optional metadata blocks
writer.write_metadata_files(
    signature=df,
    properties={"units": "ms"},
    survey={...},
    instrument={...},
    job={...}
)

# Append provenance
writer.append_provenance({"action": "processing", "details": "..."})
```

### InternalFormatReader

Load datasets (typically used via `SeismicData.open()`):

```python
from pyseis_io.core.reader import InternalFormatReader

reader = InternalFormatReader(path)
sd = reader.read()  # Returns SeismicData
```

## Validation and Error Handling

The library performs strict validation:

### File Presence
- `traces.zarr/` must exist
- `trace.parquet` must exist
- `metadata/metadata.json` must exist

### Schema Integrity
- All schema files validated against checksums
- Headers validated against schema column definitions
- Missing required columns raise `ValueError`

### Data Integrity
- Header count must match trace count
- `sample_rate` required in metadata
- Invalid slices raise appropriate exceptions

## Legacy Format Support

### SEG-Y

```python
from pyseis_io.segy import SEGYReader

reader = SEGYReader('file.sgy')
print(reader.text_header)
traces = reader.read_trace(0)
```

### SEG-D

```python
from pyseis_io.segd import SegD

reader = SegD('file.segd')
data = reader.read()
```

### GeoPackage Export

```python
from pyseis_io.gpkg import export_headers_to_gis

export_headers_to_gis(seismic_data, 'output.gpkg', format='gpkg')
```

## Package Structure

```
pyseis_io/
├── core/                  # Core internal format implementation
│   ├── dataset.py         # SeismicData and ParquetHeaderStore
│   ├── layout.py          # SeismicDatasetLayout
│   ├── reader.py          # InternalFormatReader
│   ├── writer.py          # InternalFormatWriter
│   └── schema.py          # SchemaManager
├── templates/
│   └── schemas/           # Schema definitions (YAML)
├── segy/                  # SEG-Y format support
├── segd/                  # SEG-D format support
├── su/                    # Seismic Unix format support
├── gpkg/                  # GeoPackage export
└── legacy/                # Legacy implementations
```

## Dependencies

- **numpy**: Array operations
- **pandas**: DataFrame handling
- **dask**: Lazy array operations
- **zarr**: Chunked array storage
- **pyarrow**: Parquet I/O and Arrow types
- **pyyaml**: YAML parsing
- **construct**: Binary format parsing (SEG-Y/SEG-D)
- **geopandas**: GeoPackage export (optional)

## Development

### Running Tests

```bash
# Run all core tests
pytest tests/core/

# Run specific test file
pytest tests/core/test_internal_format.py

# Run with verbose output
pytest tests/core/ -v
```

### Test Coverage

The core functionality has comprehensive unit tests covering:
- Dataset creation and overwriting
- Writing (NumPy/Dask arrays, headers, metadata)
- Reading and slicing operations
- Compute functionality
- Provenance tracking
- Schema validation
- Error handling
- Round-trip integrity

## Design Principles

1. **Lazy Access**: Traces stay on disk via Dask + Zarr until `.compute()`
2. **Schema-Driven Validation**: All metadata tables validated through YAML schemas
3. **Strict Invariants**: Required file presence enforced at read time
4. **Slice-Stable Access**: Slices preserve alignment between traces and headers
5. **Immutable Lineage**: Provenance file grows append-only

## License

GNU Affero General Public License v3.0

## Related Projects

- **PySeis**: Higher-level seismic data processing library that depends on pyseis-io

## Documentation

For detailed architecture and API documentation, see [docs/architecture.md](docs/architecture.md).
