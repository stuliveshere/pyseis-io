import pytest
import numpy as np
from pyseis_io.segd.segd import SegD

# We need a fixture for SEGD data too, but it's more complex.
# For now, let's just test the class structure if we can't easily mock the binary format without more work.
# Or we can try to mock the internal SegD21Format object.

def test_segd_reader_init():
    # We can't easily init without a real file or a mocked open
    # So we'll mock the SegD21Format class
    pass

# Since mocking is complex without a framework like unittest.mock or pytest-mock installed (though unittest is stdlib),
# and we didn't add pytest-mock to dependencies, let's try to use unittest.mock
from unittest.mock import MagicMock, patch

@patch('pyseis_io.segd.segd.SegD21Format')
def test_segd_reader_mocked(mock_format_cls):
    mock_instance = MagicMock()
    mock_format_cls.return_value = mock_instance
    
    # Setup mock data
    mock_instance.get_traces.return_value = [
        MagicMock(trace_data=[1.0, 2.0, 3.0], demux_header={'k': 'v'}, trace_header_extensions=[])
    ]
    
    reader = SegD("dummy.segd")
    
    assert reader.num_traces == 1
    assert reader.samples_per_trace == 3
    
    data = reader.get_trace_data(0)
    assert len(data) == 3
    assert data[0] == 1.0
