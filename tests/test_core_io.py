import shutil
import tempfile
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import dask.array as da

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyseis_io.core.dataset import SeismicData
from pyseis_io.core.writer import InternalFormatWriter

def test_round_trip():
    print("Starting round trip test...")
    # Create dummy data
    n_traces = 100
    n_samples = 500
    data = np.random.random((n_traces, n_samples)).astype(np.float32)
    dask_data = da.from_array(data, chunks=(10, n_samples))
    
    # Create dummy headers
    headers = pd.DataFrame({
        'trace_id': np.arange(n_traces),
        'offset': np.random.random(n_traces) * 1000,
        'source_x': np.zeros(n_traces),
        'receiver_x': np.arange(n_traces) * 10
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dataset"
        
        # 1. Write using InternalFormatWriter directly
        print("Writing dataset 1...")
        writer = InternalFormatWriter(path)
        writer.write_traces(dask_data)
        writer.write_headers(headers)
        writer.write_metadata({'sample_rate': 2000.0})
        
        # 2. Open using SeismicData.open
        print("Opening dataset 1...")
        ds = SeismicData.open(path)
        
        # 3. Verify
        print("Verifying dataset 1...")
        assert ds.n_traces == n_traces
        assert ds.n_samples == n_samples
        assert ds.sample_rate == 2000.0
        
        # Verify data
        loaded_data, loaded_headers = ds.compute()
        np.testing.assert_allclose(loaded_data, data)
        
        # Reset index for comparison as parquet might not preserve it exactly or we don't care
        pd.testing.assert_frame_equal(
            loaded_headers.reset_index(drop=True), 
            headers.reset_index(drop=True),
            check_dtype=False # Parquet might change types slightly (e.g. int32 vs int64)
        )
        
        # 4. Save to new location
        print("Saving dataset 2 (SeismicData.save)...")
        path2 = Path(tmpdir) / "test_dataset_2"
        ds.save(path2)
        
        # 5. Open second dataset
        print("Opening dataset 2...")
        ds2 = SeismicData.open(path2)
        loaded_data2, loaded_headers2 = ds2.compute()
        
        print("Verifying dataset 2...")
        np.testing.assert_allclose(loaded_data2, data)
        pd.testing.assert_frame_equal(
            loaded_headers2.reset_index(drop=True), 
            headers.reset_index(drop=True),
            check_dtype=False
        )
        
        # Clean up
        ds.close()
        ds2.close()
        
    print("Test passed!")

if __name__ == "__main__":
    test_round_trip()
