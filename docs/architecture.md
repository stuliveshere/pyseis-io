# SeismicData Architecture

This document explains the architecture of PySeis's core `SeismicData` model, which provides lazy, out-of-core access to seismic datasets.

## Overview

The `SeismicData` architecture separates concerns between:
- **Storage**: Parquet (headers) and Zarr (trace data)
- **Lazy Access**: Dask arrays and windowed Parquet reads
- **In-Memory Header Processing**: Arrow-backed Pandas DataFrames

This design enables efficient handling of TB-scale datasets that exceed available RAM.

## Architecture Diagram

```mermaid
flowchart TD

SRC[Original Dataset: SEG-Y / SEG-D]:::source

PQT[Parquet Headers: Arrow columnar storage, Row-grouped]:::storage
ZR[Zarr Data Store: Chunked trace data, Dask-friendly]:::storage

SD[SeismicData lazy view]:::seismic

HSTORE[ParquetHeaderStore: Minimal row-window reads, Column projection, Returns Arrow Tables]:::store

HEADERS[Arrow-backed Pandas DataFrame: Full pandas semantics, Boolean masks, Complex indexing]:::pandas

DASK[Dask Array: Lazy slicing, Lazy graph construction, Out-of-memory compute]:::dask

PQT_OUT[New Parquet Headers: written via pandas/Arrow]:::output
ZR_OUT[New Zarr Data Store: written via Dask compute]:::output

SRC --> PQT
SRC --> ZR

PQT --> SD
ZR --> SD

SD -->|sd.headers| HSTORE
HSTORE -->|Arrow Table| HEADERS

SD -->|sd.data lazy slice| DASK

HEADERS -->|write_parquet| PQT_OUT
DASK -->|compute/write_zarr| ZR_OUT

classDef source fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000
classDef storage fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000
classDef seismic fill:#f8cecc,stroke:#b85450,stroke-width:2px,color:#000
classDef store fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#000
classDef pandas fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000
classDef dask fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000
classDef output fill:#cfe2f3,stroke:#6fa8dc,stroke-width:2px,color:#000
```

## Key Components

### 1. Storage Layer

#### Parquet Headers
- **Format**: Arrow columnar storage with row groups
- **Purpose**: Efficient storage and retrieval of trace metadata
- **Benefits**:
  - Column projection (read only needed fields)
  - Row-group based slicing (minimal I/O)
  - Compression

#### Zarr Data Store
- **Format**: Chunked N-dimensional arrays
- **Purpose**: Store trace sample data
- **Benefits**:
  - Dask-friendly chunking
  - Parallel I/O
  - Cloud-ready (works on S3/GCS)

### 2. Lazy Access Layer

#### SeismicData
- **Role**: Lazy container managing views over data and headers
- **Key Features**:
  - Slicing creates new views without loading data
  - Internal `_trace_slice` tracks current window
  - Copy-on-write semantics

#### ParquetHeaderStore
- **Role**: Minimal window reader for Parquet files
- **API**: `read_window(start, stop, columns=None)` → Arrow Table
- **Limitation**: No fancy indexing (handled by Pandas layer)

### 3. In-Memory Processing Layer

#### Arrow-backed Pandas DataFrame
- **Created**: When `sd.headers` is accessed
- **Benefits**:
  - Full Pandas API (boolean masks, fancy indexing, etc.)
  - Zero-copy from Arrow when possible
  - Efficient memory usage

#### Dask Array
- **Created**: Wraps Zarr store
- **Benefits**:
  - Lazy slicing and computation
  - Out-of-memory algorithms
  - Parallel execution

## Usage Example

```python
import dask.array as da
from pyseis_io.models import SeismicData, ParquetHeaderStore

# Open a dataset (lazy)
data = da.from_zarr('path/to/traces.zarr')
header_store = ParquetHeaderStore('path/to/headers.parquet')
sd = SeismicData(data, header_store, sample_rate=2000.0)

# Slice (still lazy)
subset = sd[1000:2000]

# Access headers (triggers read of rows 1000-2000)
headers = subset.headers  # Returns Pandas DataFrame

# Filter using Pandas
cdp_subset = headers[headers['cdp'] > 5000]

# Access data (lazy Dask array)
trace_data = subset.data  # Still lazy

# Compute (triggers actual loading)
data_np, headers_pd = subset.compute()
```

## Design Rationale

### Why Parquet for Headers?
- Columnar format enables reading only needed fields
- Row groups allow efficient range queries
- Arrow integration provides zero-copy to Pandas
- Industry standard with broad tooling support

### Why Zarr for Data?
- Native chunking matches seismic processing patterns
- Dask integration is seamless
- Cloud-native (unlike HDF5)
- Parallel writes without locking

### Why Not Dask DataFrame for Headers?
- `dd.DataFrame.iloc` is incomplete and slow
- Partition boundaries make alignment unpredictable
- Headers are typically small enough for in-memory processing
- Pandas provides richer indexing semantics

### Separation of Concerns
- **Parquet/Arrow**: Efficient disk I/O
- **Pandas**: Rich in-memory indexing API
- **Dask**: Out-of-core computation
- **SeismicData**: Lazy view management
