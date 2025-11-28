import pytest
import numpy as np
from pyseis_io.segy.segy import SEGYReader

# Example small public SEGY file (Teapot Dome or similar is often used)
# For this test, we'll use a placeholder or a small file if we can find one.
# Since we can't guarantee internet access or specific URL stability, 
# we'll mark this test to be skipped if download fails.

TEST_SEGY_URL = "https://raw.githubusercontent.com/equinor/segyio-notebooks/master/data/small.sgy"

def test_integration_read_real_segy(download_test_file):
    try:
        filename = download_test_file(TEST_SEGY_URL, "small.sgy")
    except Exception as e:
        pytest.skip(f"Could not download test file: {e}")
        return

    reader = SEGYReader(filename)
    reader.read()
    
    assert reader.num_traces > 0
    assert reader.samples_per_trace > 0
    assert reader.sample_rate > 0
    
    # Check first trace
    trace0 = reader.get_trace_data(0)
    assert isinstance(trace0, np.ndarray)
    assert len(trace0) == reader.samples_per_trace

    # Check header
    header0 = reader.get_trace_header(0)
    assert isinstance(header0, dict)
