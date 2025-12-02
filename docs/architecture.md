# **pyseis-io: Updated Architecture & API Documentation (Latest Version)**

pyseis-io defines a **strict, validated internal format** for seismic datasets, combining:

* **Zarr** for trace samples (chunked, compressed, lazy IO)
* **Parquet** for headers (columnar, efficient slicing)
* **YAML/JSON** for structured metadata
* **Dask** for scalable lazy loading
* **Schema-driven validation** for all metadata components

The system ensures reproducibility, correctness, and efficient access for large seismic projects.

---

# 1. Dataset On-Disk Layout

Based on `SeismicDatasetLayout` , every dataset stored in pyseis-io has the following directory structure:

```
<dataset_root>/
    traces.zarr/                 # Zarr group containing 'data' array
        .zgroup
        data/                    # trace array chunks

    trace.parquet                # trace headers (required)
    source.parquet               # optional
    receiver.parquet             # optional
    signature.parquet            # optional signature attributes

    metadata/
        metadata.json            # required global metadata
        provenance.yaml          # history of operations
        schema_manifest.yaml     # checksums + installed schema metadata
        layout.yaml              # layout version metadata
        properties.yaml          # optional
        survey.yaml              # optional
        instrument.yaml          # optional
        job.yaml                 # optional

    schema/
        <component>/
            vX.Y.yaml            # installed schema definitions
```

Required items:

* `traces.zarr/`
* `trace.parquet`
* `metadata/metadata.json`
* schema directory + manifest

The layout is validated by `SeismicDatasetLayout.ensure_structure()`.

---

# 2. Core Modules and Classes

## **2.1 SeismicData**

Defined in `dataset.py` 

This is the **primary in-memory representation** of a dataset. It ties together:

* Dask trace array
* Parquet header store
* Metadata (sample rate)
* Dataset slicing logic
* Save/open operations

### Constructor

```python
SeismicData(
    data: dask.array.Array,
    header_store: ParquetHeaderStore,
    sample_rate: float,
    file_path: Optional[str] = None,
    _trace_slice: Optional[slice] = None
)
```

### Key Properties

```python
sd.n_traces       # number of traces
sd.n_samples      # trace length
sd.headers        # pandas DataFrame for current slice
sd.data           # dask array
sd.sample_rate    # float
sd.file_path      # dataset location
```

### Slicing

Slicing returns **new SeismicData views**, not arrays:

```python
sd[5]             # 1-trace view
sd[10:20]         # 10-trace view
sd[::2]           # decimated view
```

Header slicing is aligned via trace index mapping.

### Compute

```python
traces_np, headers_df = sd.compute()
```

Materializes both traces and headers into memory.

### Save

```python
sd.save("new_dataset", overwrite=True)
```

Writes:

* traces → Zarr
* `sd.headers` → trace.parquet
* metadata → metadata.json
* schemas / provenance → auto-handled by writer

### Open

```python
sd = SeismicData.open("dataset_path")
```

Delegates to `InternalFormatReader`.

---

## **2.2 ParquetHeaderStore**

Defined in `dataset.py` 

Efficient lazy reader for Parquet row subsets.

### Methods

```python
store = ParquetHeaderStore("trace.parquet")

store.read_window(start, stop)
len(store)
store.close()
```

### Features

* Precomputes row-group index for fast slicing.
* Returns Arrow tables.
* Handles negative indices and empty ranges.

---

## **2.3 SeismicDatasetLayout**

Defined in `layout.py` 

Responsible for:

* Dataset directory construction
* Schema installation (via SchemaManager)
* Layout validation and invariants
* Dataset rename/copy/delete

### Key Paths

```python
layout.traces_path              # root/traces.zarr
layout.trace_metadata_path      # root/trace.parquet
layout.global_metadata_path     # root/metadata/metadata.json
layout.provenance_path
layout.schema_dir
layout.signature_metadata_path
```

### Validation

`ensure_structure()` checks:

* Root exists
* Schemas validated
* `traces.zarr`, `trace.parquet`, `metadata.json` exist

---

## **2.4 SchemaManager**

Defined in `schema.py` 

Schema lifecycle support:

* Install schemas into dataset
* Create manifest containing version + checksums
* Validate schema integrity
* Validate DataFrames against schema column definitions

### Key Methods

```python
sm = SchemaManager(root)

sm.install()               # copies schemas from template package
sm.validate()              # integrity + checksum check
sm.validate_dataframe(df, "trace_header")
```

### Manifest Contents

* installed schemas
* checksum
* version
* timestamp
* pyseis-io version

---

## **2.5 InternalFormatWriter**

Defined in `writer.py` 

Writes datasets in the internal format.

### Construction

```python
writer = InternalFormatWriter(root, overwrite=True)
```

Automatic behavior:

* Remove dataset if `overwrite=True`
* Create new layout
* Install schemas
* Initialize provenance

### Writing Trace Data

```python
writer.write_traces(
    data,
    chunks=None,
    compressor="blosc",
    compression_level=5
)
```

Supports:

* NumPy arrays
* Dask arrays
* automatic chunk inference

Writes to:

```
root/traces.zarr/data
```

### Writing Header Tables

```python
writer.write_headers(
    trace_headers=df1,
    source_headers=df2,
    receiver_headers=df3
)
```

Includes schema validation for:

* `trace_header`
* `source`
* `receiver`

### Writing Metadata

```python
writer.write_metadata({"sample_rate": 0.002})
```

### Writing Additional Metadata Blocks

```python
writer.write_metadata_files(
    signature=df,
    properties={"units": "ms"},
    survey={...},
    instrument={...},
    job={...}
)
```

### Provenance

```python
writer.append_provenance({"action": "wrote headers"})
```

Automatically adds timestamp and user.

---

## **2.6 InternalFormatReader**

Defined in `reader.py` 

Loads a dataset into `SeismicData`.

### Behavior

* Validates layout structure and schemas
* Loads `traces.zarr/data` via Dask
* Wraps `trace.parquet` in ParquetHeaderStore
* Reads metadata.json → sample_rate
* Validates trace/header count consistency

### API

```python
reader = InternalFormatReader(path)
sd = reader.read()
```

Or user-level:

```python
sd = SeismicData.open(path)
```

---

# 3. End-to-End Workflow

## **3.1 Create + Write Dataset**

```python
import numpy as np
import pandas as pd
from pyseis_io.writer import InternalFormatWriter

data = np.random.randn(100, 2000)
headers = pd.DataFrame({
    "trace_id": np.arange(100),
    "offset": np.linspace(10, 1000, 100)
})

w = InternalFormatWriter("dataset", overwrite=True)
w.write_traces(data)
w.write_headers(trace_headers=headers)
w.write_metadata({"sample_rate": 0.002})
```

---

## **3.2 Load Dataset**

```python
from pyseis_io.dataset import SeismicData

sd = SeismicData.open("dataset")
print(sd.n_traces, sd.n_samples)
```

---

## **3.3 Slice**

```python
subset = sd[10:20]
subset.headers
subset.data     # lazy Dask array
```

---

## **3.4 Compute**

```python
traces_np, hdr_df = sd.compute()
```

---

## **3.5 Save a Derived Dataset**

```python
sd[0:50].save("subset_ds", overwrite=True)
```

---

# 4. Error Handling and Validation

The code now performs **strict validation**, including:

### 1. File presence

`ensure_structure()` checks:

* traces.zarr exists
* trace.parquet exists
* metadata.json exists

### 2. Schema integrity

`SchemaManager.validate()` verifies:

* schema_manifest.yaml exists
* all schema files exist
* checksums match

### 3. Data integrity

Reader checks header count matches trace count:

```plaintext
Header/trace count mismatch
```

### 4. Metadata presence

`sample_rate` must exist in metadata.json.

### 5. Slice errors

Invalid slicing raises appropriate exceptions.

---

# 5. Public API Summary

## Top-level object

```python
SeismicData.open(path)
SeismicData.save(path, overwrite=False)
```

## Read-only fields

```python
sd.data       # Dask trace array
sd.headers    # pandas DataFrame (Arrow-backed)
sd.sample_rate
sd.n_traces
sd.n_samples
```

## Operations

```python
sd[i]         # slicing → returns new SeismicData
sd.compute()  # materialize
sd.close()
```

## Writing API

```python
InternalFormatWriter(root, overwrite=False)

write_traces(data, chunks=None, compressor='blosc', compression_level=5)
write_headers(trace_headers, source_headers=None, receiver_headers=None)
write_metadata({...})
write_metadata_files(...)
append_provenance({...})
```

## Metadata API

```python
SchemaManager.install()
SchemaManager.validate()
SchemaManager.validate_dataframe(df, "trace_header")
```

---

# 6. Internal Design Principles

1. **Lazy access**
   Traces stay on disk via Dask + Zarr until `.compute()`.

2. **Schema-driven validation**
   All metadata tables validated through YAML schemas.

3. **Strict on-disk invariants**
   Required file presence:

   * Zarr trace data
   * Parquet trace headers
   * metadata.json
   * Provenance + schema manifest

4. **Slice-stable access**
   Slices preserve alignment between:

   * trace samples
   * trace headers

5. **Immutable lineage**
   Provenance file grows append-only.

