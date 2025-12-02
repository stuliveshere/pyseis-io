import pytest
import numpy as np
import pandas as pd
import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
from pyseis_io.models import SeismicData, ParquetHeaderStore

@pytest.fixture
def mock_parquet_file(tmp_path):
    """Create a mock Parquet file."""
    path = tmp_path / "test_headers.parquet"
    
    # Create a table with 100 rows
    df = pd.DataFrame({
        'trace_num': range(100),
        'offset': np.random.random(100) * 1000
    })
    table = pa.Table.from_pandas(df)
    
    # Write with small row groups to test slicing logic
    # 100 rows, row_group_size=10 -> 10 row groups
    pq.write_table(table, path, row_group_size=10)
    
    return str(path), df

@pytest.fixture
def mock_seismic_data(mock_parquet_file):
    """Create a mock SeismicData instance."""
    path, expected_df = mock_parquet_file
    
    n_traces = 100
    n_samples = 1000
    sample_rate = 2000.0
    
    # Create dask array
    data_np = np.random.random((n_traces, n_samples))
    data = da.from_array(data_np, chunks=(10, n_samples))
    
    # Create HeaderStore
    store = ParquetHeaderStore(path)
    
    return SeismicData(data, store, sample_rate), data_np, expected_df

def test_initialization(mock_seismic_data):
    """Test proper initialization of SeismicData."""
    sd, _, _ = mock_seismic_data
    
    assert isinstance(sd.data, da.Array)
    assert isinstance(sd.header_store, ParquetHeaderStore)
    assert sd.sample_rate == 2000.0
    assert sd.n_traces == 100
    assert sd.n_samples == 1000

def test_lazy_slicing(mock_seismic_data):
    """Test that slicing returns a new view without materializing headers."""
    sd, _, _ = mock_seismic_data
    
    # Slice
    subset = sd[10:30]
    
    assert isinstance(subset, SeismicData)
    assert subset.n_traces == 20
    # Check internal slice state
    assert subset._trace_slice == slice(10, 30, 1)
    
    # Headers should NOT be materialized yet (accessing .headers triggers it)
    # We can't easily check "not materialized" without mocking, but we verify behavior.

def test_header_materialization(mock_seismic_data):
    """Test that accessing .headers returns the correct Pandas DataFrame."""
    sd, _, expected_df = mock_seismic_data
    
    # Full headers
    headers = sd.headers
    assert isinstance(headers, pd.DataFrame)
    assert len(headers) == 100
    pd.testing.assert_frame_equal(headers.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=False)
    
    # Sliced headers
    subset = sd[10:30]
    sub_headers = subset.headers
    assert len(sub_headers) == 20
    assert sub_headers['trace_num'].iloc[0] == 10
    assert sub_headers['trace_num'].iloc[-1] == 29

def test_chained_slicing(mock_seismic_data):
    """Test that multiple slices compose correctly."""
    sd, _, _ = mock_seismic_data
    
    # Slice 10:90 -> 80 traces (indices 10..89)
    subset1 = sd[10:90]
    # Slice 5:15 relative to subset1 -> indices 15..24 (absolute)
    subset2 = subset1[5:15]
    
    assert subset2.n_traces == 10
    assert subset2._trace_slice == slice(15, 25, 1)
    
    headers = subset2.headers
    assert headers['trace_num'].iloc[0] == 15
    assert headers['trace_num'].iloc[-1] == 24

def test_compute(mock_seismic_data):
    """Test compute() returns in-memory objects."""
    sd, expected_data, expected_headers = mock_seismic_data
    
    data, headers = sd.compute()
    
    assert isinstance(data, np.ndarray)
    assert isinstance(headers, pd.DataFrame)
    np.testing.assert_array_equal(data, expected_data)
    pd.testing.assert_frame_equal(headers.reset_index(drop=True), expected_headers.reset_index(drop=True), check_dtype=False)
