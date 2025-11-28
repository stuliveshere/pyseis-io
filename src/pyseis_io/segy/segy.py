import os
from typing import Dict, Any, Optional
import numpy as np
from construct import *
from pyseis_io.utils import ibm2ieee, ieee2ibm
from pyseis_io.base import SeismicReader

class IBMFloatAdapter(Adapter):
    def _decode(self, obj, context, path):
        return ibm2ieee(np.array(obj))
    
    def _encode(self, obj, context, path):
        val = ieee2ibm(np.array(obj))
        return int(val) if np.isscalar(val) else val.astype(int).tolist()

# Add EBCDIC to ASCII conversion adapter
class EBCDICAdapter(Adapter):
    def _decode(self, obj, context, path):
        # Convert EBCDIC to ASCII
        try:
            return obj.decode('cp037')  # cp037 is the EBCDIC code page
        except:
            return obj
    
    def _encode(self, obj, context, path):
        # Convert ASCII to EBCDIC if needed
        try:
            return obj.encode('cp037')
        except:
            return obj

# SEGY format definition
segy_format = Struct(
    "ebcdic" / Bytes(3200),

    "binary" / Struct(
        "job_id" / Int32ub,
        "line_number" / Int32ub,
        "reel_number" / Int32ub,
        "number_of_data_traces_per_ensemble" / Int16ub,
        "number_of_auxiliary_traces_per_ensemble" / Int16ub,
        "sample_interval" / Int16ub,
        "sample_interval_original" / Int16ub,
        "number_of_samples_per_trace" / Int16ub,
        "number_of_samples_original" / Int16ub,
        "data_sample_format_code" / Int16ub,
        "ensemble_fold" / Int16ub,
        "trace_sorting_code" / Int16ub,
        "vertical_sum_code" / Int16ub,
        "sweep_frequency_start" / Int16ub,
        "sweep_frequency_end" / Int16ub,
        "sweep_length" / Int16ub,
        "sweep_type_code" / Int16ub,
        "trace_number_of_sweep_channel" / Int16ub,
        "sweep_trace_taper_length_start" / Int16ub,
        "sweep_trace_taper_length_end" / Int16ub,
        "taper_type" / Int16ub,
        "correlated_traces" / Int16ub,
        "binary_gain_recovered" / Int16ub,
        "amplitude_recovery_method" / Int16ub,
        "measurement_system" / Int16ub,
        "impulse_signal_polarity" / Int16ub,
        "vibratory_polarity_code" / Int16ub,
        None / Bytes(340)  # Remaining bytes for custom data
    ),

    "traces" / GreedyRange(
        Struct(
            "header" / Struct(
                "trace_sequence_number_within_line" / Int32ub,
                "trace_sequence_number_within_file" / Int32ub,
                "original_field_record_number" / Int32ub,
                "trace_number_within_field_record" / Int32ub,
                "energy_source_point" / Int32ub,
                "ensemble_number" / Int32ub,
                "trace_number_within_ensemble" / Int32ub,
                "trace_identification_code" / Int16ub,
                "number_of_vertically_summed_traces" / Int16ub,
                "number_of_horizontally_stacked_traces" / Int16ub,
                "data_use" / Int16ub,
                "source_receiver_offset" / Int32ub,
                "receiver_group_elevation" / Int32ub,
                "surface_elevation_at_source" / Int32ub,
                "source_depth_below_surface" / Int32ub,
                "datum_elevation_at_receiver_group" / Int32ub,
                "datum_elevation_at_source" / Int32ub,
                "water_depth_at_source" / Int32ub,
                "water_depth_at_receiver_group" / Int32ub,
                "scalar_for_elevations" / Int16sb,
                "scalar_for_coordinates" / Int16sb,
                "source_coordinate_x" / Int32ub,
                "source_coordinate_y" / Int32ub,
                "group_coordinate_x" / Int32ub,
                "group_coordinate_y" / Int32ub,
                "coordinate_units" / Int16ub,
                "weathering_velocity" / Int16ub,
                "subweathering_velocity" / Int16ub,
                "uphole_time_at_source" / Int16ub,
                "uphole_time_at_group" / Int16ub,
                "source_static_correction" / Int16ub,
                "group_static_correction" / Int16ub,
                "total_static_applied" / Int16ub,
                "lag_time_A" / Int16ub,
                "lag_time_B" / Int16ub,
                "delay_recording_time" / Int16ub,
                "mute_time_start" / Int16ub,
                "mute_time_end" / Int16ub,
                "number_of_samples_in_this_trace" / Int16ub,
                "sample_interval_in_this_trace" / Int16ub,
                None / Bytes(240 - 118),
            ),
            "samples" / Array(lambda this: this.header.number_of_samples_in_this_trace, IBMFloatAdapter(Int32ub)),
        )
    )
)

class SEGYReader(SeismicReader):
    def __init__(self, filename: str):
        self.filename = filename
        self.parsed_data = None
        self._read_metadata()

    def _read_metadata(self):
        """Read only the headers to initialize metadata."""
        # For now, we'll read the whole file as the current construct definition is monolithic.
        # In a real optimization, we would lazy load.
        pass

    def read(self) -> None:
        """Read the entire file."""
        with open(self.filename, 'rb') as f:
            self.parsed_data = segy_format.parse_stream(f)

    @property
    def num_traces(self) -> int:
        if self.parsed_data:
            return len(self.parsed_data.traces)
        return 0

    @property
    def samples_per_trace(self) -> int:
        if self.parsed_data and self.parsed_data.binary:
            return self.parsed_data.binary.number_of_samples_per_trace
        return 0

    @property
    def sample_rate(self) -> float:
        if self.parsed_data and self.parsed_data.binary:
            return self.parsed_data.binary.sample_interval
        return 0.0

    def get_trace_data(self, index: int) -> np.ndarray:
        if not self.parsed_data:
            self.read()
        if 0 <= index < self.num_traces:
            return np.array(self.parsed_data.traces[index].samples)
        raise IndexError(f"Trace index {index} out of range")

    def get_trace_header(self, index: int) -> Dict[str, Any]:
        if not self.parsed_data:
            self.read()
        if 0 <= index < self.num_traces:
            return dict(self.parsed_data.traces[index].header)
        raise IndexError(f"Trace index {index} out of range")

