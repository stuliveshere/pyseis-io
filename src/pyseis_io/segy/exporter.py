import os
import struct
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

from pyseis_io.core.dataset import SeismicData
from pyseis_io.base import SeismicExporter
from pyseis_io.core.format_parser import FormatParser

class SEGYExporter(SeismicExporter):
    """
    Writer for exporting internal format to SEGY files.
    Always writes IEEE Float32 (format=5).
    """
    
    def __init__(self, seismic_data: Union[SeismicData, str, Path], header_def: Optional[str] = None, mapping_path: Optional[str] = None):
        """
        Initialize the SEGY Exporter.
        
        Args:
             seismic_data: Source dataset.
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

        # Load Definition
        if header_def:
             self.header_def_path = Path(header_def)
        else:
             self.header_def_path = Path(__file__).parent / "segy_format.yaml"
             
        with open(self.header_def_path, 'r') as f:
             self.header_def = yaml.safe_load(f)
             
        self._trace_header_size = self.header_def['trace_header'].get('block_size', 240)
        self._binary_header_size = self.header_def['binary_header'].get('block_size', 400)
        self._ebcdic_header_size = self.header_def['ebcdic_header'].get('block_size', 3200)

        # Load Mapping
        if mapping_path:
             self.mapping_path = Path(mapping_path)
        else:
             self.mapping_path = Path(__file__).parent / "header_mapping.yaml"
             
        with open(self.mapping_path, 'r') as f:
             self.mapping = yaml.safe_load(f)
             
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        
        # Parsers
        self.trace_parser = FormatParser(self.header_def['trace_header'], endian='>')
        self.binary_parser = FormatParser(self.header_def['binary_header'], endian='>')

    def _create_default_ebcdic(self) -> bytes:
        """Create a default 3200-byte EBCDIC header."""
        # Simple C01-C40 structure
        text = "C01 SEG-Y Output from pyseis-io" + " " * 49 + "\n"
        for i in range(2, 41):
             text += f"C{i:02d}" + " " * 77 + "\n"
        
        # Truncate/Pad to 3200
        text = text[:3200].ljust(3200)
        try:
             return text.encode('cp500') # EBCDIC
        except LookupError:
             return text.encode('ascii') # Fallback if cp500 missing (rare)

    def _create_binary_header(self, ns: int, dt_micros: int) -> bytes:
        """Create BCD binary header."""
        # Default values
        header = {
             'jobid': 1,
             'lino': 1,
             'reno': 1,
             'ntrpr': 0,
             'nart': 0,
             'hdt': dt_micros,
             'dto': dt_micros,
             'hns': ns,
             'nso': ns,
             'format': 5, # IEEE Float32
             'fold': 1,
             'tsort': 4, # CDP sorted? or 1 (as recorded)
             'vscode': 0,
             'hsfs': 0,
             'hsfe': 0,
             'hslen': 0,
             'hstyp': 0,
             'schn': 0,
             'hstas': 0,
             'hstae': 0,
             'htatyp': 0,
             'hcorr': 0,
             'bgrcv': 0,
             'rcvm': 0,
             'mfeet': 1, # Meters
             'polyv': 0,
             'vpol': 0,
             'segyrev': 0x0100, # Rev 1.0 (256 decimal)
             'fixedlen': 0,
             'numhdr': 0,
        }
        
        # We need a pack logic for binary header too. 
        # FormatParser doesn't have a 'pack_to_bytes' yet, but we can use build_read_dtype on a dataframe row.
        # Or construct manually since it's one record.
        # Creating a 1-row DataFrame and writing via to_records/tobytes is easiest using FormatParser logic 
        # IF we had a writer-dtype. 
        
        # Let's use parser's build_read_dtype (which constructs the structure) and fill it.
        dtype = self.binary_parser.build_read_dtype(fixed_size=self._binary_header_size)
        arr = np.zeros(1, dtype=dtype)
        
        for k, v in header.items():
             if k in dtype.names:
                  arr[k] = v
                  
        return arr.tobytes()

    def export(self, output_path: Union[str, Path], **kwargs) -> None:
        """Export to SEGY."""
        output_segy_path = Path(output_path)
        
        if self._sd_instance:
             sd = self._sd_instance
        else:
             sd = SeismicData.open(self.seismic_data_path)
        
        ns = sd.n_samples
        dt_micros = int(sd.sample_rate * 1_000_000)
        n_traces = sd.n_traces
        
        with open(output_segy_path, 'wb') as f:
             # 1. EBCDIC
             f.write(self._create_default_ebcdic())
             
             # 2. Binary Header
             f.write(self._create_binary_header(ns, dt_micros))
             
             # 3. Traces
             # Composite dtype for writing
             header_dtype = self.trace_parser.build_read_dtype(fixed_size=self._trace_header_size)
             data_dtype_str = '>f4' # IEEE Big Endian
             
             composite_dtype = np.dtype([
                  ('header', header_dtype),
                  ('data', (data_dtype_str, (ns,)))
             ])
             
             chunk_size = 1000
             
             for i in range(0, n_traces, chunk_size):
                  end = min(i + chunk_size, n_traces)
                  chunk_sd = sd[i:end]
                  traces, headers = chunk_sd.compute()
                  
                  # Prepare Header DataFrame
                  segy_headers = pd.DataFrame(index=headers.index)
                  for seis_key, segy_key in self.reverse_mapping.items():
                       if seis_key in headers.columns:
                            segy_headers[segy_key] = headers[seis_key]
                            
                  # Defaults, NaNs, Types
                  for key in header_dtype.names:
                       if key not in segy_headers.columns:
                            segy_headers[key] = 0
                  
                  segy_headers = segy_headers.fillna(0)
                  
                  # Scaling logic (Scalco/Scalel) - simplified to 1 for now or preserve if mapped
                  if 'scalco' not in segy_headers.columns: segy_headers['scalco'] = 1
                  if 'scalel' not in segy_headers.columns: segy_headers['scalel'] = 1
                  
                  # Set ns, dt
                  segy_headers['ns'] = ns
                  segy_headers['dt'] = dt_micros
                  
                  # Write buffer
                  chunk_len = len(segy_headers)
                  buffer = np.zeros(chunk_len, dtype=composite_dtype)
                  
                  # Fill headers
                  for name in header_dtype.names:
                       if name in segy_headers.columns:
                            # Ensure type correctness for numpy assignment
                            buffer['header'][name] = segy_headers[name].values.astype(int) # mostly ints
                  
                  # Fill data
                  buffer['data'] = traces.astype('>f4')
                  
                  buffer.tofile(f)
                  
        print(f"SEGY Export complete: {output_segy_path}")
