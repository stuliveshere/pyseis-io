import pytest
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path so pyseis_io can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# from construct import Container

from pathlib import Path
from pyseis_io.segy.segy import segy_format
from pyseis_io.segd.formats.segd21 import (
    general_header1,
    general_header2,
    general_header_n,
    channel_set_header,
    demux_header,
    trace_header_extension_format
)
from tests.utils import download_file

@pytest.fixture
def mock_segy_data():
    """Create a mock SEGY file structure in memory."""
    def _create_segy(num_traces=1, samples_per_trace=10, sample_interval=1000, header_overrides=None):
        # 3200 bytes EBCDIC
        ebcdic = b' ' * 3200
        
        # Binary Header
        binary_header_data = {
            'job_id': 1,
            'line_number': 1,
            'reel_number': 1,
            'number_of_data_traces_per_ensemble': 1,
            'number_of_auxiliary_traces_per_ensemble': 0,
            'sample_interval': sample_interval,
            'sample_interval_original': sample_interval,
            'number_of_samples_per_trace': samples_per_trace,
            'number_of_samples_original': samples_per_trace,
            'data_sample_format_code': 5, # IEEE Float
            'ensemble_fold': 1,
            'trace_sorting_code': 1,
            'vertical_sum_code': 1,
            'sweep_frequency_start': 0,
            'sweep_frequency_end': 0,
            'sweep_length': 0,
            'sweep_type_code': 0,
            'trace_number_of_sweep_channel': 0,
            'sweep_trace_taper_length_start': 0,
            'sweep_trace_taper_length_end': 0,
            'taper_type': 0,
            'correlated_traces': 0,
            'binary_gain_recovered': 0,
            'amplitude_recovery_method': 0,
            'measurement_system': 1,
            'impulse_signal_polarity': 1,
            'vibratory_polarity_code': 0,
            None: b'\x00' * 340
        }
        
        if header_overrides and 'binary' in header_overrides:
            binary_header_data.update(header_overrides['binary'])
        
        traces = []
        for i in range(num_traces):
            trace_header_data = {
                'trace_sequence_number_within_line': i + 1,
                'trace_sequence_number_within_file': i + 1,
                'original_field_record_number': 1,
                'trace_number_within_field_record': i + 1,
                'energy_source_point': 1,
                'ensemble_number': 1,
                'trace_number_within_ensemble': i + 1,
                'trace_identification_code': 1,
                'number_of_vertically_summed_traces': 1,
                'number_of_horizontally_stacked_traces': 1,
                'data_use': 1,
                'source_receiver_offset': 100,
                'receiver_group_elevation': 0,
                'surface_elevation_at_source': 0,
                'source_depth_below_surface': 0,
                'datum_elevation_at_receiver_group': 0,
                'datum_elevation_at_source': 0,
                'water_depth_at_source': 0,
                'water_depth_at_receiver_group': 0,
                'scalar_for_elevations': 1,
                'scalar_for_coordinates': 1,
                'source_coordinate_x': 0,
                'source_coordinate_y': 0,
                'group_coordinate_x': 100,
                'group_coordinate_y': 0,
                'coordinate_units': 1,
                'weathering_velocity': 0,
                'subweathering_velocity': 0,
                'uphole_time_at_source': 0,
                'uphole_time_at_group': 0,
                'source_static_correction': 0,
                'group_static_correction': 0,
                'total_static_applied': 0,
                'lag_time_A': 0,
                'lag_time_B': 0,
                'delay_recording_time': 0,
                'mute_time_start': 0,
                'mute_time_end': 0,
                'number_of_samples_in_this_trace': samples_per_trace,
                'sample_interval_in_this_trace': sample_interval,
                None: b'\x00' * (240 - 118)
            }
            
            if header_overrides and 'trace' in header_overrides:
                trace_header_data.update(header_overrides['trace'])
                
            samples = [float(x) for x in range(samples_per_trace)]
            traces.append({'header': trace_header_data, 'samples': samples})
            
        data = {
            'ebcdic': ebcdic,
            'binary': binary_header_data,
            'traces': traces
        }
        
        return segy_format.build(data)
    return _create_segy

@pytest.fixture
def temp_segy_file(tmp_path, mock_segy_data):
    """Create a temporary SEGY file."""
    p = tmp_path / "test.sgy"
    p.write_bytes(mock_segy_data())
    return str(p)

@pytest.fixture
def mock_segd_data():
    """Create a mock SEG-D file structure in memory."""
    def _create_segd(num_traces=1, samples_per_trace=10):
        # Build components individually and concatenate
        
        # General Header 1
        gh1_data = {
            "file_number": 1,
            "format_code": 8058, # IEEE Float
            "year": 80,
            "additional_gh_blocks": 0, # No extra blocks for simplicity
            "day": 1,
            "hour": 0,
            "minute": 0,
            "second": 0,
            "manufacturer_code": 61, # SmartSolo (as per segd21.py default)
            "manufacturer_serial": 1,
            "base_scan_interval_raw": 16, # 1.0 ms
            "polarity": 0,
            "record_type": 8,
            "record_length": 1000, # ms
            "scan_types_per_record": 1,
            "channel_sets_per_scan_type": 1,
            "skew_blocks": 0,
            "extended_header_blocks": 0,
            "external_header_blocks": 0
        }
        gh1_bytes = general_header1.build(gh1_data)
        
        # Channel Set Header
        csh_data = {
            "scan_type_number": 1,
            "channel_set_number": 1,
            "start_time_raw": 0,
            "end_time_raw": 1000, # 2000ms / 2
            "descale_multiplier_extended": 0,
            "descale_multiplier_raw": 0,
            "number_of_channels": num_traces,
            "channel_type_raw": 1, # Seismic
            "subscans": 0,
            "gain_control_method": 1,
            "alias_filter_freq": 0,
            "alias_filter_slope": 0,
            "low_cut_freq": 0,
            "low_cut_slope": 0,
            "notch_freq1": 0,
            "notch_freq2": 0,
            "notch_freq3": 0,
            "extended_channel_set_number": 0,
            "extended_header_flag": 0,
            "trace_header_extensions": 1, # 1 extension
            "vertical_stack": 1,
            "streamer_cable_number": 0,
            "array_forming": 0
        }
        csh_bytes = channel_set_header.build(csh_data)
        
        # Traces
        trace_bytes = b""
        for i in range(num_traces):
            # Demux Header
            demux_data = {
                "file_number": 1,
                "scan_type_number": 1,
                "channel_set_number": 1,
                "trace_number": i + 1,
                "first_timing_word": 0,
                "trace_header_extensions": 1,
                "sample_skew": 0,
                "trace_edit": 0,
                "time_break_window_int": 0,
                "time_break_window_frac": 0,
                "extended_channel_set_number": 0,
                "extended_file_number": 0
            }
            trace_bytes += demux_header.build(demux_data)
            
            # Trace Header Extension 1
            ext1_data = {
                "receiver_line_number": 1,
                "receiver_point_number": i + 1,
                "receiver_point_index": 1,
                "samples_per_trace": samples_per_trace,
                "extended_receiver_line_number_int": 0,
                "extended_receiver_line_number_frac": 0.0,
                "extended_receiver_point_number_int": 0,
                "extended_receiver_point_number_frac": 0.0,
                "sensor_type": 2, # Geophone Vertical
                "undefined": b'\x00' * 11
            }
            trace_bytes += trace_header_extension_format.build(ext1_data)
            
            # Trace Data (IEEE Float)
            # 4 bytes per sample
            samples = np.array([float(x) for x in range(samples_per_trace)], dtype=np.float32)
            trace_bytes += samples.tobytes()
            
        return gh1_bytes + csh_bytes + trace_bytes
    return _create_segd

@pytest.fixture
def temp_segd_file(tmp_path, mock_segd_data):
    """Create a temporary SEG-D file."""
    p = tmp_path / "test.segd"
    p.write_bytes(mock_segd_data())
    return str(p)

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Directory to store downloaded test data."""
    return tmp_path_factory.mktemp("data")

@pytest.fixture
def download_test_file(test_data_dir):
    """Fixture to download a test file."""
    def _download(url, filename=None):
        return str(download_file(url, test_data_dir, filename))
    return _download
