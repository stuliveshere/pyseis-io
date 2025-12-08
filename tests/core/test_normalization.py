
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from pyseis_io.core.writer import InternalFormatWriter
from pyseis_io.core.dataset import SeismicData
from pyseis_io.core.layout import SeismicDatasetLayout

def test_normalization_valid_flow(tmp_path):
    """Test valid normalization of source/receiver tables."""
    seis_path = tmp_path / "valid.seis"
    
    # Create Flat DataFrame
    data = {
        'trace_id': [1, 2, 3],
        'source_id': ['S1', 'S1', 'S2'],
        'receiver_id': ['R1', 'R2', 'R2'],
        # Source attributes
        'source_x': [100.0, 100.0, 200.0],
        'source_y': [10.0, 10.0, 20.0],
        # Receiver attributes
        'receiver_x': [500.0, 600.0, 600.0],
        # Trace attributes
        'offset': [400.0, 500.0, 400.0]
    }
    df = pd.DataFrame(data)
    
    # Write (will create layout)
    writer = InternalFormatWriter(seis_path, overwrite=True)
    writer.write_headers(df)
    
    # Verify Files Exist
    assert (seis_path / "source.parquet").exists()
    assert (seis_path / "receiver.parquet").exists()
    assert (seis_path / "trace.parquet").exists()
    
    # Verify Content
    source_df = pd.read_parquet(seis_path / "source.parquet")
    assert len(source_df) == 2
    assert source_df.iloc[0]['source_id'] == 'S1'
    assert 'source_x' in source_df.columns
    assert 'offset' not in source_df.columns
    
    trace_df = pd.read_parquet(seis_path / "trace.parquet")
    assert 'source_x' not in trace_df.columns
    assert 'offset' in trace_df.columns
    assert 'source_id' in trace_df.columns # FK exists

def test_join_on_read(tmp_path):
    """Test that SeismicData joins headers automatically."""
    seis_path = tmp_path / "join.seis"
    
    data = {
        'trace_id': [1],
        'source_id': ['S1'],
        'receiver_id': ['R1'],
        'source_x': [100.0],
        'receiver_x': [500.0],
        'offset': [400.0]
    }
    df = pd.DataFrame(data)
    
    writer = InternalFormatWriter(seis_path, overwrite=True)
    writer.write_headers(df)
    
    # Write Dummy Data to open SeismicData
    writer.write_traces(np.zeros((1, 10)))
    writer.write_metadata({'sample_rate': 0.004})
    
    # Open
    sd = SeismicData.open(seis_path)
    headers = sd.headers
    
    # Check Columns Present (Joined)
    assert 'source_x' in headers.columns
    assert 'receiver_x' in headers.columns
    assert 'offset' in headers.columns
    
    # Check Values
    assert headers.iloc[0]['source_x'] == 100.0
    
    sd.close()

def test_conflict_validation(tmp_path):
    """Test error on conflicting ID attributes."""
    seis_path = tmp_path / "conflict.seis"
    
    # Conflict: S1 has x=100 and x=101
    data = {
        'trace_id': [1, 2],
        'source_id': ['S1', 'S1'],
        'source_x': [100.0, 101.0], 
        'offset': [10, 20]
    }
    df = pd.DataFrame(data)
    
    writer = InternalFormatWriter(seis_path, overwrite=True)
    
    with pytest.raises(ValueError, match="Source ID conflict detected"):
        writer.write_headers(df)

def test_summary_output(tmp_path):
    """Test that summary() runs and mentions the tables."""
    seis_path = tmp_path / "summary.seis"
    
    data = {
        'trace_id': [1],
        'source_id': ['S1'],
        'receiver_id': ['R1'],
        'source_x': [100.0],
        'receiver_x': [500.0],
        'offset': [400.0]
    }
    df = pd.DataFrame(data)
    
    writer = InternalFormatWriter(seis_path, overwrite=True)
    writer.write_headers(df)
    writer.write_traces(np.zeros((1, 10)))
    writer.write_metadata({'sample_rate': 0.004})
    
    sd = SeismicData.open(seis_path)
    s = sd.summary()
    
    assert "Trace Headers (trace.parquet)" in s
    assert "Source Attributes (source.parquet)" in s
    assert "Receiver Attributes (receiver.parquet)" in s
    assert "source_x" in s
    assert "offset" in s
    
    sd.close()

def test_metadata_extraction(tmp_path):
    """Test that constant globals are moved to metadata.json."""
    seis_path = tmp_path / "meta_extract.seis"
    
    data = {
        'trace_id': [1, 2],
        'source_id': ['S1', 'S1'], # Same source
        'receiver_id': ['R1', 'R2'],
        'sample_rate': [0.004, 0.004], # Constant
        'coordinate_scalar': [-100, -100], # Constant
        'elevation_scalar': [-100, -1000], # Varying (should remain in trace/receiver)
        'receiver_x': [100.0, 200.0]
    }
    df = pd.DataFrame(data)
    
    writer = InternalFormatWriter(seis_path, overwrite=True)
    writer.write_headers(df) # Should extract sample_rate and coordinate_scalar
    writer.write_traces(np.zeros((2, 10))) # Write dummy traces
    
    # Check metadata.json
    import json
    with open(writer.layout.global_metadata_path, 'r') as f:
        meta = json.load(f)
        
    assert meta['sample_rate'] == 0.004
    assert meta['coordinate_scalar'] == -100
    assert 'elevation_scalar' not in meta # It varies
    
    # Check tables
    sd = SeismicData.open(seis_path)
    
    # trace.parquet should NOT have sample_rate or coordinate_scalar
    # But it MIGHT have elevation_scalar if it wasn't in source/receiver schema (removed from receiver)
    # verify headers
    h = sd.headers
    assert 'elevation_scalar' in h.columns
    # extracted globals are NOT in columns via join, they are in metadata
    # Unless sd.headers logic injects them? (No, it doesn't currently)
    # The user wanted them in metadata.json.
    # Current SeismicData doesn't inject metadata into .headers view automatically?
    # TODO: Maybe it should? But for now verifying extraction.
    
    sd.close()
