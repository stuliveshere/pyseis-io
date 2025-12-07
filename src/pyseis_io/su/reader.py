import os
import struct
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from pyseis_io.core.writer import InternalFormatWriter

# Mapping SU headers to SeisData schema
# Key: SU header name (from su.yaml) -> Value: SeisData column name
SU_TO_SEISDATA = {
    # Trace related
    "tracl": "trace_id", "tracr": "trace_sequence_number", "offset": "offset", 
    "tstat": "total_static", "trid": "trace_identification_code",
    "cdp": "cdp_id", "muts": "mute_start", "mute": "mute_end",
    "corr": "correlated", "trwf": "trace_weighting_factor",
    # Data related
    "ns": "num_samples", "dt": "sample_rate", "delrt": "recording_delay",
    # Coordinate system
    "scalco": "coordinate_scalar", "scalel": "elevation_scalar",
    # Source related
    "fldr": "source_id", "ep": "source_index", "sx": "source_x", 
    "sy": "source_y", "sdepth": "source_z", "sut": "uphole_time",
    # Receiver related
    "tracf": "receiver_index", "gx": "receiver_x", 
    "gy": "receiver_y", "gelev": "receiver_z",
}

class SUConverter:
    """
    Reader for Seismic Unix files that converts to pyseis-io internal format.
    Supports 'Scan -> Modify -> Convert' workflow.
    """
    
    def __init__(self, su_path: Union[str, Path], header_def: Optional[str] = None):
        """
        Initialize the SU Converter.
        
        Args:
            su_path: Path to the SU file.
            header_def: Optional path to a YAML file defining the SU header structure.
                        Defaults to src/pyseis_io/su/su.yaml.
        """
        self.su_path = Path(su_path)
        if not self.su_path.exists():
            raise FileNotFoundError(f"SU file not found: {su_path}")
            
        # Load header definition
        if header_def:
            self.header_def_path = Path(header_def)
        else:
            self.header_def_path = Path(__file__).parent / "su.yaml"
            
        if not self.header_def_path.exists():
             raise FileNotFoundError(f"Header definition file not found: {self.header_def_path}")

        with open(self.header_def_path, 'r') as f:
            self.header_def = yaml.safe_load(f)
            
        if 'SU_TRACE_HEADER' not in self.header_def or 'definition' not in self.header_def['SU_TRACE_HEADER']:
             raise ValueError("Invalid header definition format.")
             
        self.su_headers = self.header_def['SU_TRACE_HEADER']['definition']
        
        # Verify required keys in custom definition
        required_keys = [k for k in SU_TO_SEISDATA.keys()]
        # We check if keys used in mapping exist in definition
        for key in required_keys:
             if key not in self.su_headers:
                  # Warning or Error? Let's error if critical.
                  # But some files might miss some headers.
                  # The converter logic needs these keys to map.
                  # If the key isn't in definition, we can't read it.
                  # However, the USER might provide a custom definition that renames/excludes things.
                  # For now, require keys in SU_TO_SEISDATA to be present in definition
                  # if we want to map them.
                  pass
        
        # Internal state
        self._endian = None
        self._ns = None
        self._dt = None
        self._file_size = self.su_path.stat().st_size
        self.headers: Optional[pd.DataFrame] = None
        self._initial_trace_count = 0
        self._trace_stride = 0

    def _detect_endianness(self, f) -> str:
        """
        Detect endianness by checking ns and dt sanity.
        """
        # Read first 240 bytes
        f.seek(0)
        header_bytes = f.read(240)
        if len(header_bytes) < 240:
             raise ValueError("File too short for header")
             
        # Definition for ns and dt
        ns_def = self.su_headers.get('ns')
        dt_def = self.su_headers.get('dt')
        
        if not ns_def or not dt_def:
             raise ValueError("ns and dt must be defined in header definition")
             
        # Helper to read value
        def read_val(bytes_data, def_, endian):
             start = def_['start_byte']
             n_bytes = def_['num_bytes']
             fmt = self._get_fmt_char(def_['format'], n_bytes)
             return struct.unpack(f'{endian}{fmt}', bytes_data[start:start+n_bytes])[0]
             
        # Check Little Endian
        ns_le = read_val(header_bytes, ns_def, '<')
        dt_le = read_val(header_bytes, dt_def, '<')
        
        # Check Big Endian
        ns_be = read_val(header_bytes, ns_def, '>')
        dt_be = read_val(header_bytes, dt_def, '>')
        
        # Heuristics
        # ns should be positive and reasonable (e.g. < 100,000)
        # dt is usually in microseconds (e.g. 1000, 2000, 4000)
        
        is_le_valid = 0 < ns_le < 100000 and 0 <= dt_le < 1000000
        is_be_valid = 0 < ns_be < 100000 and 0 <= dt_be < 1000000
        
        # refinement using file size
        file_size = self.su_path.stat().st_size
        if is_le_valid:
            stride_le = 240 + ns_le * 4
            if file_size % stride_le != 0 and file_size > stride_le: # Allow truncated if just scanning? No, scan assumes structure.
                 # If file is perfectly valid size for BE but not LE, that's a strong hint.
                 # But real files might be truncated.
                 # Let's give a "score" or strictly prefer the one that fits.
                 pass
        
        if is_le_valid and is_be_valid:
             # Check consistency
             stride_le = 240 + ns_le * 4
             stride_be = 240 + ns_be * 4
             
             consistent_le = (file_size % stride_le == 0)
             consistent_be = (file_size % stride_be == 0)
             
             if consistent_le and not consistent_be:
                 return '<'
             if consistent_be and not consistent_le:
                 return '>'
                 
             # Still ambiguous? Default to LE with warning.
             print("Warning: Endianness ambiguous (both valid and size-consistent), defaulting to Little Endian.")
             return '<'

        if is_le_valid and not is_be_valid:
             return '<'
        if is_be_valid and not is_le_valid:
             return '>'
             
        raise ValueError("Could not detect valid endianness.")

    def _get_fmt_char(self, fmt_type, n_bytes):
        """Map SU format/bytes to struct format character."""
        if fmt_type == 'uint':
            if n_bytes == 2: return 'H'
            if n_bytes == 4: return 'I'
        elif fmt_type == 'int':
            if n_bytes == 2: return 'h'
            if n_bytes == 4: return 'i'
        elif fmt_type == 'float':
            if n_bytes == 4: return 'f'
            if n_bytes == 8: return 'd'
        
        # Fallback
        if n_bytes == 2: return 'h'
        if n_bytes == 4: return 'i'
        return 'i' # Default

    def scan(self) -> pd.DataFrame:
        """
        Scan all headers from the SU file into a DataFrame.
        This reads ALL headers efficiently without reading trace data.
        
        Returns:
            pd.DataFrame: The headers dataframe.
        """
        with open(self.su_path, 'rb') as f:
            # 1. Detect Endianness & Geometry
            self._endian = self._detect_endianness(f)
            
            # Read first header to get ns for stride
            f.seek(0)
            header_bytes = f.read(240)
            ns_def = self.su_headers['ns']
            dt_def = self.su_headers['dt']
            
            start_ns = ns_def['start_byte']
            fmt_ns = self._get_fmt_char(ns_def['format'], ns_def['num_bytes'])
            self._ns = struct.unpack(f'{self._endian}{fmt_ns}', header_bytes[start_ns:start_ns+ns_def['num_bytes']])[0]
            
            start_dt = dt_def['start_byte']
            fmt_dt = self._get_fmt_char(dt_def['format'], dt_def['num_bytes'])
            self._dt = struct.unpack(f'{self._endian}{fmt_dt}', header_bytes[start_dt:start_dt+dt_def['num_bytes']])[0]
            
            # Calculate stride
            # Header (240) + Data (ns * 4 bytes for float32)
            self._trace_stride = 240 + self._ns * 4
            
            # Calculate number of traces
            if self._file_size % self._trace_stride != 0:
                print(f"Warning: File size {self._file_size} is not a multiple of trace stride {self._trace_stride}. Truncated/corrupt file?")
            
            self._initial_trace_count = self._file_size // self._trace_stride
            
            # 2. Memory Map & Stride Tricks
            # We want to view the file as an array of traces, but only read the first 240 bytes of each trace.
            mmap = np.memmap(self.su_path, dtype='uint8', mode='r')
            
            # Ensure we don't go out of bounds if file is truncated
            cutoff = self._initial_trace_count * self._trace_stride
            if cutoff > len(mmap):
                 mmap = mmap[:cutoff]
            
            # Create a strided view of headers: (n_traces, 240)
            # shape=(n_traces, 240)
            # strides=(trace_stride, 1)
            headers_view = np.lib.stride_tricks.as_strided(
                mmap, 
                shape=(self._initial_trace_count, 240), 
                strides=(self._trace_stride, 1)
            )
            
            # 3. Extract Columns
            data_dict = {}
            for key, def_ in self.su_headers.items():
                start = def_['start_byte']
                n_bytes = def_['num_bytes']
                fmt = def_['format']
                
                # Extract byte slice for this column across all traces
                # View: (n_traces, n_bytes)
                col_view = headers_view[:, start:start+n_bytes]
                
                # Convert to numpy array of correct type
                # We need to copy to contiguous memory to view as type
                # Optimization: Can we avoid copy? struct.unpack needs bytes. 
                # np.frombuffer needs contiguous.
                # headers_view is strided, so rows are far apart.
                # But within a row (header), bytes are contiguous.
                # However, col_view[:, :] is not contiguous in memory as a block.
                
                # Efficient way:
                # np.ndarray(buffer=mmap, dtype=..., offset=start, strides=trace_stride)
                # This works if endianness matches native.
                
                # Map SU types to numpy dtypes
                np_dtype = None
                if fmt in ['int', 'uint']:
                    if n_bytes == 2: np_dtype = np.int16 if fmt=='int' else np.uint16
                    elif n_bytes == 4: np_dtype = np.int32 if fmt=='int' else np.uint32
                elif fmt == 'float':
                    if n_bytes == 4: np_dtype = np.float32
                    elif n_bytes == 8: np_dtype = np.float64
                
                if np_dtype:
                    # Create strided view for this column
                    # Note: offset is absolute in file.
                    # start is relative to trace start.
                    # We start at 'start' byte of file.
                    col_array = np.ndarray(
                        shape=(self._initial_trace_count,),
                        dtype=np_dtype,
                        buffer=mmap,
                        offset=start,
                        strides=(self._trace_stride,)
                    )
                    
                    # Handle Endianness
                    # If file endianness != system endianness, byteswap
                    sys_endian = '<' if np.little_endian else '>'
                    if self._endian != sys_endian:
                        data_dict[key] = col_array.byteswap()
                    else:
                        data_dict[key] = col_array.copy() # Copy to decouple from mmap
                
            # Create DataFrame
            self.headers = pd.DataFrame(data_dict)
            
            # 4. Map & Scale
            self._map_and_scale()
            
            return self.headers

    def _map_and_scale(self):
        """Map SU headers to SeisData and apply defaults."""
        if self.headers is None: return
        
        # Renaissance mapping
        renamed_df = self.headers.rename(columns=SU_TO_SEISDATA)
        
        # Apply Scalars
        if 'coordinate_scalar' in renamed_df.columns:
            coords = ['source_x', 'source_y', 'receiver_x', 'receiver_y']
            for col in coords:
                if col in renamed_df.columns:
                    renamed_df[col] = self._apply_scalar(renamed_df[col], renamed_df['coordinate_scalar'])
                    
        if 'elevation_scalar' in renamed_df.columns:
            elev_cols = ['source_z', 'receiver_z'] # mapped in SU_TO_SEISDATA
            for col in elev_cols:
                if col in renamed_df.columns:
                    renamed_df[col] = self._apply_scalar(renamed_df[col], renamed_df['elevation_scalar'])

        # Type Conversions
        # String IDs
        for col in ['source_id', 'receiver_id', 'cdp_id']:
            if col in renamed_df.columns:
                renamed_df[col] = renamed_df[col].astype(str)
        
        # Special case for receiver_id if overwritten
        if 'receiver_id' not in renamed_df.columns and 'trace_sequence_number' in renamed_df.columns:
             renamed_df['receiver_id'] = renamed_df['trace_sequence_number'].astype(str)
             
        # Bool
        if 'correlated' in renamed_df.columns:
            renamed_df['correlated'] = renamed_df['correlated'].astype(bool)
            
        # Defaults
        if 'trace_weighting_factor' not in renamed_df.columns:
            renamed_df['trace_weighting_factor'] = 1.0
            
        if 'trace_identification_code' not in renamed_df.columns:
            renamed_df['trace_identification_code'] = 1
            
        self.headers = renamed_df

    def _apply_scalar(self, series, scalar_series):
        """Apply SU scalar logic."""
        # copy to float
        out = series.astype(float)
        
        # Mask for scal < 0 (divisor)
        mask_div = scalar_series < 0
        div_val = scalar_series.abs()
        # Avoid division by zero
        div_val[div_val == 0] = 1 
        
        out[mask_div] = out[mask_div] / div_val[mask_div]
        
        # Mask for scal > 0 (multiplier)
        mask_mul = scalar_series > 0
        out[mask_mul] = out[mask_mul] * scalar_series[mask_mul]
        
        return out

    def convert(self, output_path: Union[str, Path]):
        """
        Convert the SU file to internal format using the (possibly modified) headers.
        """
        if self.headers is None:
            raise RuntimeError("Headers not loaded. Call scan() first.")
            
        # Validation checks
        if len(self.headers) != self._initial_trace_count:
             raise ValueError(f"Header count mismatch: Expected {self._initial_trace_count}, got {len(self.headers)}. Dropping traces is not supported.")
             
        # Check index integrity (optional validation, assuming user didn't reset index)
        if not self.headers.index.equals(pd.RangeIndex(self._initial_trace_count)):
             # If index was specialized, we can't easily check order without a stable ID.
             # But we can assume data is row-aligned.
             # Warning if index is suspicious?
             pass
             
        output_path = Path(output_path)
        
        # Prepare Metadata
        # Use first trace sample rate
        sample_rate = self._dt / 1_000_000.0 # micros -> seconds
        
        # Initialize Writer
        writer = InternalFormatWriter(output_path, overwrite=True)
        writer.write_metadata({"sample_rate": sample_rate})
        
        # Write Headers
        # Ensure only valid schema columns are written? Writer handles validation.
        # But we need to ensure we pass a DataFrame compatible with schema.
        # SUReader produces 'trace_sequence_number', 'sample_rate' etc.
        writer.write_headers(self.headers)
        
        # Write Traces
        # Iterate and write in chunks
        chunk_size = 1000
        
        with open(self.su_path, 'rb') as f:
            for i in range(0, self._initial_trace_count, chunk_size):
                count = min(chunk_size, self._initial_trace_count - i)
                
                # Pre-allocate array
                traces = np.zeros((count, self._ns), dtype=np.float32)
                
                for j in range(count):
                    trace_idx = i + j
                    # Calculate offset
                    offset = trace_idx * self._trace_stride + 240 # Skip header
                    
                    f.seek(offset)
                    data_bytes = f.read(self._ns * 4)
                    
                    # Unpack
                    # Efficient: np.frombuffer
                    trace_data = np.frombuffer(data_bytes, dtype=np.float32)
                    
                    # Handle byteswap
                    sys_endian = '<' if np.little_endian else '>'
                    if self._endian != sys_endian:
                        trace_data = trace_data.byteswap()
                        
                    traces[j, :] = trace_data
                    
                writer.write_traces(traces)
                
        print(f"Conversion complete: {output_path}")
