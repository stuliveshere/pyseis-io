import pytest
import numpy as np
from pyseis_io.segy.segy import SEGYReader

def test_segy_reader_init(temp_segy_file):
    reader = SEGYReader(temp_segy_file)
    assert reader.filename == temp_segy_file

def test_segy_reader_read(temp_segy_file):
    reader = SEGYReader(temp_segy_file)
    reader.read()
    assert reader.num_traces == 1
    assert reader.samples_per_trace == 10
    assert reader.sample_rate == 1000

def test_segy_get_trace_data(temp_segy_file):
    reader = SEGYReader(temp_segy_file)
    data = reader.get_trace_data(0)
    assert isinstance(data, np.ndarray)
    assert len(data) == 10
    assert data[0] == 0.0

def test_segy_get_trace_header(temp_segy_file):
    reader = SEGYReader(temp_segy_file)
    header = reader.get_trace_header(0)
    assert header['trace_sequence_number_within_line'] == 1
    assert header['number_of_samples_in_this_trace'] == 10
