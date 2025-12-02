import pytest
import numpy as np
from pyseis_io.segd.segd import SegD

# Placeholder URL for SEGD file
# Finding a small public SEGD file is harder than SEGY.
# This is a placeholder for when a suitable file is identified.
TEST_SEGD_URL = "https://example.com/sample.segd" 

def test_integration_read_real_segd(download_test_file):
    try:
        filename = download_test_file(TEST_SEGD_URL, "sample.segd")
    except Exception as e:
        pytest.skip(f"Could not download test file: {e}")
        return

    # This might fail if the file is not a valid SEGD or if the reader has issues
    # But the test structure is here.
    try:
        reader = SegD(filename)
        reader.read()
        
        assert reader.num_traces > 0
        assert reader.samples_per_trace > 0
        
        # Check first trace
        trace0 = reader.get_trace_data(0)
        assert isinstance(trace0, np.ndarray)
        
    except Exception as e:
        pytest.fail(f"Failed to read SEGD file: {e}")
