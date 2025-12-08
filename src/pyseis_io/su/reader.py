import os
import struct
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple


from pyseis_io.core.writer import InternalFormatWriter
from pyseis_io.base import SeismicImporter

# Mapping SU headers to SeisData schema
# Key: SU header name (from su.yaml) -> Value: SeisData column name
class SUImporter(SeismicImporter):
    """
    Reader for Seismic Unix files that converts to pyseis-io internal format.
    Supports 'Scan -> Modify -> Convert' workflow.
    """
    
    def __init__(self, path: Union[str, Path], header_def: Optional[str] = None, mapping_path: Optional[str] = None):
        """
        Initialize the SU Converter.
        
        Args:
            path: Path to the SU file.
            header_def: Optional path to a YAML file defining the SU header structure.
                        Defaults to src/pyseis_io/su/su.yaml.
            mapping_path: Optional path to a YAML file defining SU->Core mapping.
                          Defaults to src/pyseis_io/su/mapping.yaml.
        """
        self.su_path = Path(path)
        if not self.su_path.exists():
            raise FileNotFoundError(f"SU file not found: {path}")
            
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
        
        # Load Mapping
        if mapping_path:
             self.mapping_path = Path(mapping_path)
        else:
             self.mapping_path = Path(__file__).parent / "mapping.yaml"
             
        if not self.mapping_path.exists():
             raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
             
        with open(self.mapping_path, 'r') as f:
             self.mapping = yaml.safe_load(f)

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
            if file_size % stride_le != 0 and file_size > stride_le: 
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
            pd.DataFrame: The headers dataframe (with SU column names).
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
            mmap = np.memmap(self.su_path, dtype='uint8', mode='r')
            cutoff = self._initial_trace_count * self._trace_stride
            if cutoff > len(mmap):
                 mmap = mmap[:cutoff]
            
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
                
                col_view = headers_view[:, start:start+n_bytes]
                
                np_dtype = None
                if fmt in ['int', 'uint']:
                    if n_bytes == 2: np_dtype = np.int16 if fmt=='int' else np.uint16
                    elif n_bytes == 4: np_dtype = np.int32 if fmt=='int' else np.uint32
                elif fmt == 'float':
                    if n_bytes == 4: np_dtype = np.float32
                    elif n_bytes == 8: np_dtype = np.float64
                
                if np_dtype:
                    col_array = np.ndarray(
                        shape=(self._initial_trace_count,),
                        dtype=np_dtype,
                        buffer=mmap,
                        offset=start,
                        strides=(self._trace_stride,)
                    )
                    
                    sys_endian = '<' if np.little_endian else '>'
                    if self._endian != sys_endian:
                        data_dict[key] = col_array.byteswap()
                    else:
                        data_dict[key] = col_array.copy()
                
            # Create DataFrame
            self.headers = pd.DataFrame(data_dict)
            
            # 4. Scale (Apply SU-specific scaling BEFORE mapping)
            self._apply_scalars_raw()
            
            return self.headers

    def _apply_scalars_raw(self):
        """Apply SU scalar logic to raw SU columns."""
        if self.headers is None: return
        
        df = self.headers
        
        # Coordinate Scalar (scalco)
        if 'scalco' in df.columns:
            # scalco applies to sx, sy, gx, gy
            coords = ['sx', 'sy', 'gx', 'gy']
            for col in coords:
                if col in df.columns:
                    df[col] = self._apply_scalar(df[col], df['scalco'])
                    
        # Elevation Scalar (scalel)
        if 'scalel' in df.columns:
            # scalel applies to gelev, selev, sdepth, gdel, sdel, swdep, gwdep
            elevs = ['gelev', 'selev', 'sdepth', 'gdel', 'sdel', 'swdep', 'gwdep']
            for col in elevs:
                if col in df.columns:
                    df[col] = self._apply_scalar(df[col], df['scalel'])

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


    def import_data(self, output_path: Union[str, Path], chunk_size: int = 1000, **kwargs) -> 'SeismicData':
        """
        Convert the SU file to internal format using the (possibly modified) headers.
        
        Args:
            output_path: Destination path for .seis dataset.
            chunk_size: Number of traces to process at a time. Default 1000.
        """
        if self.headers is None:
            raise RuntimeError("Headers not loaded. Call scan() first.")
            
        # Validation checks
        if len(self.headers) != self._initial_trace_count:
             raise ValueError(f"Header count mismatch: Expected {self._initial_trace_count}, got {len(self.headers)}. Dropping traces is not supported.")
             
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
        writer.write_headers(self.headers, mapping=self.mapping)
        
        # Initialize Trace Data
        # We know total shape from _initial_trace_count and _ns
        total_shape = (self._initial_trace_count, self._ns)
        
        # Optimize chunking: Zarr chunks should be aligned with our write chunks if possible for speed
        writer.initialize_data(
            shape=total_shape,
            chunks=(chunk_size, self._ns),
            dtype=np.float32
        )
        
        # Write Traces
        with open(self.su_path, 'rb') as f:
            for i in range(0, self._initial_trace_count, chunk_size):
                count = min(chunk_size, self._initial_trace_count - i)
                
                # Pre-allocate array
                traces = np.zeros((count, self._ns), dtype=np.float32)
                
                # Optimize read loop if needed, but this is readable
                for j in range(count):
                    trace_idx = i + j
                    # Calculate offset
                    offset = trace_idx * self._trace_stride + 240 # Skip header
                    
                    f.seek(offset)
                    data_bytes = f.read(self._ns * 4)
                    
                    # Unpack
                    trace_data = np.frombuffer(data_bytes, dtype=np.float32)
                    
                    # Handle byteswap
                    sys_endian = '<' if np.little_endian else '>'
                    if self._endian != sys_endian:
                        trace_data = trace_data.byteswap()
                        
                    traces[j, :] = trace_data
                    
                writer.write_data_chunk(traces, start_trace=i)
                
        print(f"Conversion complete: {output_path}")
        
        # Return opened dataset
        from pyseis_io.core.dataset import SeismicData
        return SeismicData.open(output_path)

