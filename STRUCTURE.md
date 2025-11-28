# pyseis-io Package Structure

After reorganization, the `pyseis_io` package has the following structure:

## Core Components

### Data Models
- `models.py` - Core data models (`SeismicData`, `ParquetHeaderStore`)
- `seisdata/` - Legacy SeisData implementation with YAML schema

### Base Classes
- `base.py` - Base reader/writer interfaces

### Utilities
- `utils.py` - Utility functions (IBM/IEEE conversion, etc.)
- `maps.py` - Header mapping definitions between formats

## Format Modules

### Active Formats
- `segy/` - SEG-Y format support
- `su/` - Seismic Unix format support
- `javaseis/` - JavaSeis format support
- `rsf/` - Madagascar RSF format support
- `gpkg/` - GeoPackage export support
- `segd/` - SEG-D format (wrapper to legacy implementation)

### Legacy Components (To Be Replaced)
- `legacy/segd_construct/` - Construct-based SEG-D reader
- `legacy/segd_yaml/` - YAML-driven SEG-D reader

## Recommended Next Steps

1. **Consolidate Data Models**: Decide between `models.py` (SeismicData) and `seisdata/` (SeisData)
2. **Clean up segd/**: Currently just a shim to legacy code
3. **Review RSF**: Verify if this format is still needed
4. **Consider structure**: Group by read/write operations or keep format-based?
