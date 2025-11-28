# pyseis-io

A standalone I/O library for seismic data, providing readers and writers for multiple industry-standard formats.

## Overview

`pyseis-io` is a foundational library extracted from PySeis that handles all file format I/O operations for seismic data. It provides a clean, stable interface for reading and writing seismic data in various formats, independent of higher-level processing tools.

## Features

- **Multiple Format Support**: Read and write SEG-Y, SEG-D, SU, JavaSeis, and export to GeoPackage
- **Unified Data Model**: Consistent `SeismicData` interface across all formats
- **Lazy Loading**: Efficient handling of large datasets with Dask integration
- **Header Mapping**: Automatic conversion between format-specific headers and unified schema
- **Endian Handling**: Automatic IBM/IEEE floating-point conversion

## Supported Formats

### Readers/Writers
- **SEG-Y**: Industry-standard seismic data format
- **SEG-D**: Field recording format (legacy implementation)
- **Seismic Unix (SU)**: Open-source seismic processing format
- **JavaSeis**: High-performance seismic data storage

### Exporters
- **GeoPackage (.gpkg)**: Export navigation data for GIS applications

## Installation

```bash
pip install -e .
```

## Quick Start

### Reading SEG-Y Data

```python
from pyseis_io.segy import SEGYReader

# Open a SEG-Y file
reader = SEGYReader('path/to/file.sgy')

# Access headers and traces
print(reader.text_header)
print(reader.binary_header)
traces = reader.read_trace(0)
```

### Reading SEG-D Data

```python
from pyseis_io.segd import SegD

# Open SEG-D file
reader = SegD('path/to/file.segd')
data = reader.read()
```

### Exporting to GeoPackage

```python
from pyseis_io.gpkg import export_headers_to_gis
from pyseis_io.models import SeismicData

# Export navigation data
export_headers_to_gis(seismic_data, 'output.gpkg', format='gpkg')
```

## Package Structure

```
pyseis_io/
├── models.py          # Core data models (SeismicData, ParquetHeaderStore)
├── base.py            # Base reader/writer interfaces
├── utils.py           # Utility functions (IBM/IEEE conversion)
├── maps.py            # Header mapping definitions
├── segy/              # SEG-Y format support
├── segd/              # SEG-D format support (wrapper)
├── su/                # Seismic Unix format support
├── javaseis/          # JavaSeis format support
├── gpkg/              # GeoPackage export support
├── rsf/               # Madagascar RSF format support
├── seisdata/          # Legacy SeisData implementation
└── legacy/            # Legacy implementations (for reference)
    ├── segd_construct/
    └── segd_yaml/
```

## Data Models

### SeismicData

Modern lazy-loading container for seismic data:

```python
from pyseis_io.models import SeismicData

# Access data and headers
n_traces = data.n_traces
n_samples = data.n_samples
headers_df = data.headers  # Pandas DataFrame
trace_data = data.data.compute()  # Dask array

# Slice data (lazy)
subset = data[0:100]
```

### SeisData

Legacy implementation with schema-based metadata:

```python
from pyseis_io.seisdata import SeisData

# Initialize with schema
data = SeisData('path/to/schema.yaml')
```

## Dependencies

- numpy
- pandas
- dask
- pyarrow
- pyyaml
- construct
- geopandas (for GeoPackage export)

## Development

### Running Tests

```bash
pytest
```

### Project Status

This library was recently extracted from PySeis to establish a stable, reusable I/O layer. Some components are marked as "legacy" and will be refactored or replaced in future versions.

## License

MIT License

## Related Projects

- **PySeis**: Higher-level seismic data processing library that depends on pyseis-io
