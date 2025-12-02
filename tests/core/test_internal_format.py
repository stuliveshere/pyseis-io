import pytest
import numpy as np
import pandas as pd
import dask.array as da
import shutil
import json
import yaml
from pathlib import Path
from typing import Generator

from pyseis_io.core.layout import SeismicDatasetLayout
from pyseis_io.core.writer import InternalFormatWriter
from pyseis_io.core.reader import InternalFormatReader
from pyseis_io.core.dataset import SeismicData

# Fixtures

@pytest.fixture
def temp_dataset_path(tmp_path: Path) -> Path:
    """Returns a path to a temporary dataset directory."""
    return tmp_path / "test_dataset"

@pytest.fixture
def populated_dataset_path(temp_dataset_path: Path) -> Generator[Path, None, None]:
    """Creates a basic populated dataset for reading tests."""
    path = temp_dataset_path
    writer = InternalFormatWriter(path, overwrite=True)
    
    # Write traces
    n_traces, n_samples = 10, 50
    data = np.zeros((n_traces, n_samples), dtype=np.float32)
    writer.write_traces(data)
    
    # Write headers (minimal schema compliant)
    headers = pd.DataFrame({
        'trace_id': np.arange(n_traces, dtype=np.int32),
        'source_id': [str(i) for i in range(n_traces)],
        'receiver_id': [str(i) for i in range(n_traces)],
        'cdp_id': [str(i) for i in range(n_traces)],
        'trace_sequence_number': np.arange(n_traces, dtype=np.int32),
        'offset': np.zeros(n_traces, dtype=np.float32),
        'mute_start': np.zeros(n_traces, dtype=np.float32),
        'mute_end': np.zeros(n_traces, dtype=np.float32),
        'total_static': np.zeros(n_traces, dtype=np.float32),
        'trace_identification_code': np.ones(n_traces, dtype=np.int32),
        'correlated': np.zeros(n_traces, dtype=bool),
        'trace_weighting_factor': np.ones(n_traces, dtype=np.float32)
    })
    writer.write_headers(headers)
    
    # Write metadata
    writer.write_metadata({'sample_rate': 1000.0})
    
    yield path
    
    if path.exists():
        SeismicDatasetLayout.delete(path)

# 1. Dataset Creation

def test_create_empty_dataset(temp_dataset_path: Path):
    """Test creating an empty dataset structure."""
    layout = SeismicDatasetLayout.create(temp_dataset_path)
    
    assert layout.provenance_path.exists()
    assert (layout.metadata_dir / "schema_manifest.yaml").exists()
    assert layout.schema_dir.exists()
    # Verify specific schema exists (assuming trace_header v1.0 is standard)
    assert (layout.schema_dir / "trace_header" / "v1.0.yaml").exists()
    
    # Ensure NO data files exist yet
    assert not layout.traces_path.exists()
    assert not layout.trace_metadata_path.exists()
    assert not layout.global_metadata_path.exists()
    
    # Verify provenance content
    with open(layout.provenance_path, 'r') as f:
        prov = yaml.safe_load(f)
    assert len(prov['history']) == 1
    assert prov['history'][0]['action'] == 'created'

def test_overwrite_existing_dataset(temp_dataset_path: Path):
    """Test overwriting an existing dataset."""
    # Create first version
    layout = SeismicDatasetLayout.create(temp_dataset_path)
    # Add a dummy file to simulate content
    (temp_dataset_path / "dummy.txt").touch()
    
    # Overwrite
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    # Check old content is gone
    assert not (temp_dataset_path / "dummy.txt").exists()
    
    # Check fresh provenance
    with open(writer.layout.provenance_path, 'r') as f:
        prov = yaml.safe_load(f)
    assert len(prov['history']) == 1
    assert prov['history'][0]['action'] == 'created'
    
    # Check no data files yet
    assert not writer.layout.traces_path.exists()

# 2. Writing Synthetic Data

def test_write_synthetic_traces_numpy(temp_dataset_path: Path):
    """Test writing NumPy array traces."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    n_traces, n_samples = 10, 50
    data = np.random.random((n_traces, n_samples)).astype(np.float32)
    
    writer.write_traces(data)
    
    assert writer.layout.traces_path.exists()
    z = da.from_zarr(str(writer.layout.traces_path), component='data')
    assert z.shape == (n_traces, n_samples)
    assert z.dtype == np.float32
    # Check chunking (default logic: min(1000, n_traces))
    assert z.chunksize == (10, 50)

def test_write_synthetic_traces_dask(temp_dataset_path: Path):
    """Test writing Dask array traces."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    n_traces, n_samples = 10, 50
    data = da.zeros((n_traces, n_samples), chunks=(5, 50), dtype=np.float32)
    
    writer.write_traces(data)
    
    z = da.from_zarr(str(writer.layout.traces_path), component='data')
    assert z.shape == (n_traces, n_samples)
    assert z.chunksize == (5, 50) # Should preserve chunks if passed as dask array

def test_write_trace_headers(temp_dataset_path: Path):
    """Test writing compliant trace headers."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    n_traces = 5
    # Minimal compliant headers based on schema
    headers = pd.DataFrame({
        'trace_id': np.arange(n_traces, dtype=np.int32),
        'source_id': ['0'] * n_traces,
        'receiver_id': ['0'] * n_traces,
        'cdp_id': ['0'] * n_traces,
        'trace_sequence_number': np.arange(n_traces, dtype=np.int32),
        'offset': np.zeros(n_traces, dtype=np.float32),
        'mute_start': np.zeros(n_traces, dtype=np.float32),
        'mute_end': np.zeros(n_traces, dtype=np.float32),
        'total_static': np.zeros(n_traces, dtype=np.float32),
        'trace_identification_code': np.ones(n_traces, dtype=np.int32),
        'correlated': np.zeros(n_traces, dtype=bool),
        'trace_weighting_factor': np.ones(n_traces, dtype=np.float32)
    })
    
    writer.write_headers(headers)
    
    assert writer.layout.trace_metadata_path.exists()
    read_headers = pd.read_parquet(writer.layout.trace_metadata_path)
    assert len(read_headers) == n_traces
    assert all(col in read_headers.columns for col in headers.columns)

def test_write_metadata(temp_dataset_path: Path):
    """Test writing global metadata."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    meta = {'sample_rate': 2000.0, 'domain': 'time'}
    
    writer.write_metadata(meta)
    
    assert writer.layout.global_metadata_path.exists()
    with open(writer.layout.global_metadata_path, 'r') as f:
        loaded = json.load(f)
    assert loaded['sample_rate'] == 2000.0
    assert loaded['domain'] == 'time'

def test_write_optional_metadata_blocks(temp_dataset_path: Path):
    """Test writing optional metadata files."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    props = {'project': 'test_project'}
    survey = {'crs': 'EPSG:4326'}
    
    writer.write_metadata_files(properties=props, survey=survey)
    
    assert writer.layout.properties_metadata_path.exists()
    assert writer.layout.survey_metadata_path.exists()
    assert not writer.layout.instrument_metadata_path.exists() # Not written
    
    with open(writer.layout.properties_metadata_path, 'r') as f:
        assert yaml.safe_load(f) == props

# 3. Reading Data Back

def test_read_full_dataset(populated_dataset_path: Path):
    """Test reading the full dataset."""
    reader = InternalFormatReader(populated_dataset_path)
    ds = reader.read()
    
    assert isinstance(ds, SeismicData)
    assert ds.n_traces == 10
    assert ds.n_samples == 50
    assert ds.sample_rate == 1000.0
    assert len(ds.headers) == 10

def test_read_by_trace_slicing(populated_dataset_path: Path):
    """Test reading by various slice types."""
    ds = SeismicData.open(populated_dataset_path)
    try:
        # Single index
        trace0 = ds[0]
        assert trace0.data.shape == (1, 50)
        assert len(trace0.headers) == 1
        
        # Slice
        slice1 = ds[3:8]
        assert slice1.data.shape == (5, 50)
        assert len(slice1.headers) == 5
        assert slice1.headers.iloc[0]['trace_id'] == 3
        
        # Step slice
        slice2 = ds[::2]
        assert slice2.data.shape == (5, 50)
        assert len(slice2.headers) == 5
        assert slice2.headers.iloc[1]['trace_id'] == 2
        
        # Negative slice
        slice3 = ds[-3:]
        assert slice3.data.shape == (3, 50)
        assert slice3.headers.iloc[0]['trace_id'] == 7
        
        # Negative range
        slice4 = ds[-5:-1]
        assert slice4.data.shape == (4, 50)
        assert slice4.headers.iloc[0]['trace_id'] == 5
    finally:
        ds.close()

def test_slice_semantics(populated_dataset_path: Path):
    """Test that slicing preserves SeismicData semantics."""
    ds = SeismicData.open(populated_dataset_path)
    ds_slice = ds[2:5]
    
    assert ds_slice.file_path == ds.file_path
    assert ds_slice.sample_rate == ds.sample_rate
    assert ds_slice.n_traces == 3
    assert ds_slice.n_samples == 50

def test_invalid_slices(populated_dataset_path: Path):
    """Test invalid slice operations."""
    ds = SeismicData.open(populated_dataset_path)
    try:
        with pytest.raises(IndexError):
            _ = ds[100] # Out of bounds
            
        with pytest.raises((TypeError, KeyError, IndexError)):
            _ = ds["invalid"]
            
        with pytest.raises(TypeError):
            _ = ds[1.5]
    finally:
        ds.close()

# 4. compute() Functionality

def test_compute_materialize_entire(populated_dataset_path: Path):
    """Test materializing the entire dataset."""
    ds = SeismicData.open(populated_dataset_path)
    data, headers = ds.compute()
    
    assert isinstance(data, np.ndarray)
    assert isinstance(headers, pd.DataFrame)
    assert data.shape == (10, 50)
    assert len(headers) == 10

def test_compute_materialize_sliced(populated_dataset_path: Path):
    """Test materializing a sliced dataset."""
    ds = SeismicData.open(populated_dataset_path)
    ds_slice = ds[2:5]
    data, headers = ds_slice.compute()
    
    assert data.shape == (3, 50)
    assert len(headers) == 3
    assert headers.iloc[0]['trace_id'] == 2

# 5. Provenance Tracking

def test_append_provenance(temp_dataset_path: Path):
    """Test appending provenance events."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    event = {'action': 'processing', 'details': 'filter'}
    writer.append_provenance(event)
    
    with open(writer.layout.provenance_path, 'r') as f:
        prov = yaml.safe_load(f)
    
    assert len(prov['history']) == 2 # created + processing
    assert prov['history'][1]['action'] == 'processing'
    assert 'timestamp' in prov['history'][1]
    assert 'user' in prov['history'][1]

def test_provenance_multiple_events(temp_dataset_path: Path):
    """Test appending multiple events preserves order."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    writer.append_provenance({'action': 'step1'})
    writer.append_provenance({'action': 'step2'})
    
    with open(writer.layout.provenance_path, 'r') as f:
        prov = yaml.safe_load(f)
        
    history = prov['history']
    assert len(history) == 3
    assert history[1]['action'] == 'step1'
    assert history[2]['action'] == 'step2'

# 6. Schema Installation and Validation

def test_schema_integrity(temp_dataset_path: Path):
    """Test schema integrity validation."""
    layout = SeismicDatasetLayout.create(temp_dataset_path)
    
    # Modify a schema file
    schema_file = layout.schema_dir / "trace_header" / "v1.0.yaml"
    with open(schema_file, 'a') as f:
        f.write("\n# corruption")
        
    # Validate should fail
    from pyseis_io.core.schema import SchemaManager
    manager = SchemaManager(temp_dataset_path)
    
    with pytest.raises(ValueError, match="integrity check failed"):
        manager.validate()

def test_header_schema_validation_negative(temp_dataset_path: Path):
    """Test writing headers with missing columns fails."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    # Missing required columns (e.g., trace_id)
    bad_headers = pd.DataFrame({'offset': [0.0]})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        writer.write_headers(bad_headers)

def test_validate_dataframe_none(temp_dataset_path: Path):
    """Test validation with None input."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    from pyseis_io.core.schema import SchemaManager
    manager = SchemaManager(temp_dataset_path)
    
    # Should not raise
    manager.validate_dataframe(None, "source")

# 7. Error Handling Tests

@pytest.mark.parametrize("missing_file", ["traces.zarr", "trace.parquet", "metadata.json"])
def test_missing_required_files(populated_dataset_path: Path, missing_file: str):
    """Test error when required files are missing."""
    target = populated_dataset_path
    if missing_file == "traces.zarr":
        target = target / "traces.zarr"
        shutil.rmtree(target)
    elif missing_file == "trace.parquet":
        target = target / "trace.parquet"
        target.unlink()
    else:
        target = target / "metadata" / "metadata.json"
        target.unlink()
        
    with pytest.raises(FileNotFoundError, match="Dataset is incomplete or corrupted"):
        InternalFormatReader(populated_dataset_path).read()

def test_header_trace_count_mismatch(temp_dataset_path: Path):
    """Test mismatch between trace count and header count."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    # 10 traces
    writer.write_traces(np.zeros((10, 50)))
    
    # 5 headers
    headers = pd.DataFrame({
        'trace_id': np.arange(5, dtype=np.int32),
        'source_id': ['0']*5, 'receiver_id': ['0']*5, 'cdp_id': ['0']*5,
        'trace_sequence_number': np.arange(5, dtype=np.int32),
        'offset': np.zeros(5), 'mute_start': np.zeros(5), 'mute_end': np.zeros(5),
        'total_static': np.zeros(5), 'trace_identification_code': np.ones(5, dtype=np.int32),
        'correlated': np.zeros(5, dtype=bool), 'trace_weighting_factor': np.ones(5)
    })
    writer.write_headers(headers)
    writer.write_metadata({'sample_rate': 1000.0})
    
    with pytest.raises(ValueError, match="Header/trace count mismatch"):
        InternalFormatReader(temp_dataset_path).read()

def test_unsupported_zarr_format(temp_dataset_path: Path):
    """Test reading Zarr without 'data' component."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    # Manually create bad zarr
    import zarr
    store = zarr.DirectoryStore(str(writer.layout.traces_path))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset('wrong_name', shape=(10, 50))
    
    # Write valid headers/meta to pass other checks
    headers = pd.DataFrame({
        'trace_id': np.arange(10, dtype=np.int32),
        'source_id': ['0']*10, 'receiver_id': ['0']*10, 'cdp_id': ['0']*10,
        'trace_sequence_number': np.arange(10, dtype=np.int32),
        'offset': np.zeros(10), 'mute_start': np.zeros(10), 'mute_end': np.zeros(10),
        'total_static': np.zeros(10), 'trace_identification_code': np.ones(10, dtype=np.int32),
        'correlated': np.zeros(10, dtype=bool), 'trace_weighting_factor': np.ones(10)
    })
    writer.write_headers(headers)
    writer.write_metadata({'sample_rate': 1000.0})
    
    # Should fail when accessing data
    # Note: da.from_zarr might not fail until compute, or might fail on init if path invalid
    # The reader init does: da.from_zarr(..., component='data')
    # This usually succeeds lazily but might fail if group structure is checked.
    # If dask is lazy, we might need to try to compute to see error, 
    # OR the reader might check existence.
    # Let's see what happens. If dask is fully lazy, we might need to compute.
    
    # Let's see what happens. If dask is fully lazy, we might need to compute.
    
    ds = InternalFormatReader(temp_dataset_path).read()
    try:
        with pytest.raises(Exception): # Dask/Zarr error (KeyError, AttributeError, etc.)
            ds.data.compute()
    finally:
        ds.close()

def test_missing_sample_rate(temp_dataset_path: Path):
    """Test missing sample_rate in metadata."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    writer.write_traces(np.zeros((1, 50)))
    
    headers = pd.DataFrame({
        'trace_id': [0], 'source_id': ['0'], 'receiver_id': ['0'], 'cdp_id': ['0'],
        'trace_sequence_number': [0], 'offset': [0.0], 'mute_start': [0.0], 'mute_end': [0.0],
        'total_static': [0.0], 'trace_identification_code': [1],
        'correlated': [False], 'trace_weighting_factor': [1.0]
    })
    writer.write_headers(headers)
    
    # Write metadata without sample_rate
    with open(writer.layout.global_metadata_path, 'w') as f:
        json.dump({'domain': 'time'}, f)
        
    with pytest.raises(ValueError, match="sample_rate is required"):
        InternalFormatReader(temp_dataset_path).read()

# 8. Resource Management

def test_resource_cleanup(populated_dataset_path: Path):
    """Test resource cleanup on close."""
    ds = SeismicData.open(populated_dataset_path)
    ds.close()
    
    # Accessing headers should fail (as store is closed/None)
    # The exact error depends on implementation, likely AttributeError or similar
    with pytest.raises(Exception):
        _ = len(ds.headers)
        
    # Double close should be safe
    ds.close()

# 9. Round-Trip Integrity Tests

def test_full_end_to_end(temp_dataset_path: Path):
    """Full end-to-end round trip test."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    n_traces, n_samples = 10, 100
    data = np.random.random((n_traces, n_samples)).astype(np.float32)
    
    headers = pd.DataFrame({
        'trace_id': np.arange(n_traces, dtype=np.int32),
        'source_id': [str(i) for i in range(n_traces)],
        'receiver_id': [str(i) for i in range(n_traces)],
        'cdp_id': [str(i) for i in range(n_traces)],
        'trace_sequence_number': np.arange(n_traces, dtype=np.int32),
        'offset': np.random.random(n_traces).astype(np.float32),
        'mute_start': np.zeros(n_traces, dtype=np.float32),
        'mute_end': np.zeros(n_traces, dtype=np.float32),
        'total_static': np.zeros(n_traces, dtype=np.float32),
        'trace_identification_code': np.ones(n_traces, dtype=np.int32),
        'correlated': np.zeros(n_traces, dtype=bool),
        'trace_weighting_factor': np.ones(n_traces, dtype=np.float32)
    })
    
    writer.write_traces(data)
    writer.write_headers(headers)
    writer.write_metadata({'sample_rate': 500.0})
    
    # Read back
    ds = SeismicData.open(temp_dataset_path)
    r_data, r_headers = ds.compute()
    
    try:
        np.testing.assert_allclose(r_data, data)
        pd.testing.assert_frame_equal(
            r_headers.reset_index(drop=True), 
            headers.reset_index(drop=True),
            check_dtype=False # Parquet/Arrow backends may change types
        )
        assert ds.sample_rate == 500.0
    finally:
        ds.close()

def test_save_valid_dataset(populated_dataset_path: Path, tmp_path: Path):
    """Test that SeismicData.save produces a valid dataset."""
    ds = SeismicData.open(populated_dataset_path)
    
    save_path = tmp_path / "saved_dataset"
    ds.save(save_path)
    
    # Verify structure
    layout = SeismicDatasetLayout(save_path)
    layout.ensure_structure() # Should pass
    
    # Verify content
    ds2 = SeismicData.open(save_path)
    assert ds2.n_traces == ds.n_traces
    assert ds2.n_samples == ds.n_samples

def test_parquet_overwrite(temp_dataset_path: Path):
    """Test that writing headers overwrites existing Parquet files."""
    writer = InternalFormatWriter(temp_dataset_path, overwrite=True)
    
    # Write first set
    headers1 = pd.DataFrame({
        'trace_id': np.arange(5, dtype=np.int32),
        'source_id': ['0']*5, 'receiver_id': ['0']*5, 'cdp_id': ['0']*5,
        'trace_sequence_number': np.arange(5, dtype=np.int32),
        'offset': np.zeros(5, dtype=np.float32),
        'mute_start': np.zeros(5, dtype=np.float32),
        'mute_end': np.zeros(5, dtype=np.float32),
        'total_static': np.zeros(5, dtype=np.float32),
        'trace_identification_code': np.ones(5, dtype=np.int32),
        'correlated': np.zeros(5, dtype=bool),
        'trace_weighting_factor': np.ones(5, dtype=np.float32)
    })
    writer.write_headers(headers1)
    
    # Write second set (different content)
    headers2 = headers1.copy()
    headers2['offset'] = 100.0
    writer.write_headers(headers2)
    
    # Read back
    read_headers = pd.read_parquet(writer.layout.trace_metadata_path)
    assert len(read_headers) == 5
    assert np.all(read_headers['offset'] == 100.0)
