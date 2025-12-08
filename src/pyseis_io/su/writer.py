"""
SU Writer for exporting pyseis-io internal format to Seismic Unix files.
"""

import os
import struct
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

from pyseis_io.core.dataset import SeismicData
from pyseis_io.base import SeismicExporter

class SUExporter(SeismicExporter):
    """
    Writer for exporting internal format to SU files.
    """
    
    def __init__(self, seismic_data: Union[SeismicData, str, Path], header_def: Optional[str] = None):
        """
        Initialize the SU Writer.
        
        Args:
            seismic_data: SeismicData object or path to the input pyseis-io dataset.
            header_def: Optional path to a YAML file defining the SU header structure.
                        Defaults to src/pyseis_io/su/su.yaml.
        """
        if isinstance(seismic_data, (str, Path)):
             self.seismic_data_path = Path(seismic_data)
             if not self.seismic_data_path.exists():
                 raise FileNotFoundError(f"Seismic dataset not found: {seismic_data}")
             self._sd_instance = None
        elif isinstance(seismic_data, SeismicData):
             self._sd_instance = seismic_data
             self.seismic_data_path = seismic_data.file_path
        else:
             raise ValueError("seismic_data must be a Path/str or SeismicData object")
            
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
        mapping_path = Path(__file__).parent / "mapping.yaml"
        if not mapping_path.exists():
             raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
             
        with open(mapping_path, 'r') as f:
             self.mapping = yaml.safe_load(f)
             
        # Create reverse mapping: Core -> SU
        # If duplicated values in SU->Core, we lose one.
        # e.g. "tracr": "trace_sequence_number" and "foo": "trace_sequence_number".
        # We need to pick one. The mapping file should be 1-to-1 ideally for round trip.
        # Or prioritized.
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def export(self, output_path: Union[str, Path], endian: str = '<', **kwargs) -> None:
        """
        Export to SU file.
        
        Args:
            output_path: Path to the output SU file.
            endian: Endianness for output ('<' for little, '>' for big).
        """
        output_su_path = Path(output_path)
        
        # Open dataset if not already
        if self._sd_instance:
             sd = self._sd_instance
        else:
             sd = SeismicData.open(self.seismic_data_path)
        
        # Get headers and data
        # We need to compute() to get everything in memory?
        # Or iterate.
        # For simplicity, let's compute() if it fits in memory, or iterate if we implement chunking.
        # The plan didn't specify chunking for writer either.
        # Let's try to be memory efficient by iterating if possible, but SeismicData slicing returns new objects.
        
        # Let's read in chunks of traces.
        chunk_size = 1000
        n_traces = sd.n_traces
        
        with open(output_su_path, 'wb') as f:
            for i in range(0, n_traces, chunk_size):
                end = min(i + chunk_size, n_traces)
                # Slice
                chunk_sd = sd[i:end]
                
                # Compute to get data and headers
                traces, headers = chunk_sd.compute()
                
                # Process headers
                # 1. Reverse map columns
                # We need to create a new dataframe with SU keys.
                su_headers_df = pd.DataFrame(index=headers.index)
                
                for seis_key, su_key in self.reverse_mapping.items():
                    if seis_key in headers.columns:
                        su_headers_df[su_key] = headers[seis_key]
                
                # Type conversions for SU format (string/bool -> int)
                # source_id, receiver_id, cdp_id -> int
                # Note: This assumes IDs are numeric strings. Non-numeric will fail or become 0/NaN.
                for su_key in ['fldr', 'tracr', 'cdp']:
                    if su_key in su_headers_df.columns:
                         # Force numeric, coerce errors to NaN then fill 0
                         su_headers_df[su_key] = pd.to_numeric(su_headers_df[su_key], errors='coerce').fillna(0).astype(int)

                # correlated -> int (0/1)
                if 'corr' in su_headers_df.columns:
                    su_headers_df['corr'] = su_headers_df['corr'].astype(int)

                # 2. Handle Scalars (scalco, scalel)
                # If we have float coordinates, we need to set scalco and convert to integer.
                # Simple logic: if any coordinate is float, set scalco to -100 or -1000?
                # Or check if they are integers.
                
                # Check coordinates
                coords = ['sx', 'sy', 'gx', 'gy']
                needs_scaling = False
                for c in coords:
                    if c in su_headers_df.columns:
                        if not pd.api.types.is_integer_dtype(su_headers_df[c]):
                            needs_scaling = True
                            break
                
                if needs_scaling:
                    # Set scalco to -100 (precision 0.01) or -1000 (0.001)
                    # Let's use -1000 for good precision
                    scalar = -1000
                    su_headers_df['scalco'] = scalar
                    for c in coords:
                        if c in su_headers_df.columns:
                            su_headers_df[c] = (su_headers_df[c] * abs(scalar)).fillna(0).astype(int)
                else:
                    su_headers_df['scalco'] = 1
                    
                # Same for elevations
                elevs = ['gelev', 'selev', 'sdepth', 'gdel', 'sdel', 'swdep', 'gwdep']
                needs_elev_scaling = False
                for c in elevs:
                    if c in su_headers_df.columns:
                         if not pd.api.types.is_integer_dtype(su_headers_df[c]):
                            needs_elev_scaling = True
                            break
                            
                if needs_elev_scaling:
                    scalar = -1000
                    su_headers_df['scalel'] = scalar
                    for c in elevs:
                        if c in su_headers_df.columns:
                            su_headers_df[c] = (su_headers_df[c] * abs(scalar)).fillna(0).astype(int)
                else:
                    su_headers_df['scalel'] = 1

                # 3. Set ns and dt
                # ns is trace length
                ns = traces.shape[1]
                su_headers_df['ns'] = ns
                
                # dt is sample rate in microseconds
                # sd.sample_rate is in seconds
                dt_micros = int(sd.sample_rate * 1_000_000)
                su_headers_df['dt'] = dt_micros
                
                # 4. Fill missing keys with 0
                for key in self.su_headers:
                    if key not in su_headers_df.columns:
                        su_headers_df[key] = 0
                        
                # 5. Write traces
                for j in range(len(su_headers_df)):
                    # Write header
                    header_bytes = self._pack_header(su_headers_df.iloc[j], endian)
                    f.write(header_bytes)
                    
                    # Write data
                    trace_data = traces[j].astype(f'{endian}f4')
                    f.write(trace_data.tobytes())
                    
        print(f"Export complete: {output_su_path}")

    def _pack_header(self, row, endian) -> bytes:
        """Pack a header row into bytes."""
        buffer = bytearray(240)
        for key, def_ in self.su_headers.items():
            start = def_['start_byte']
            n_bytes = def_['num_bytes']
            fmt_char = self._get_fmt_char(def_['format'], n_bytes)
            
            val = row[key]
            
            # Ensure type matches format
            if def_['format'] in ['uint', 'int']:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = 0
            elif def_['format'] == 'float':
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
            
            # Ensure value fits in type
            # e.g. clip or warn?
            # struct.pack will raise error if out of range for integer
            
            try:
                packed = struct.pack(f'{endian}{fmt_char}', val)
                buffer[start:start+n_bytes] = packed
            except struct.error:
                # Fallback to 0 or max?
                # print(f"Warning: Value {val} for {key} out of range for {fmt_char}")
                pass
                
        return bytes(buffer)

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
        
        if n_bytes == 2: return 'h'
        if n_bytes == 4: return 'i'
        return 'i'
