"""
Writer for internal seismic data format.
"""

import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
import zarr
import numpy as np
import pandas as pd
import dask.array as da

from .layout import SeismicDatasetLayout

class InternalFormatWriter:
    """
    Writer for seismic data in the internal Zarr/Parquet format.
    
    This writer converts seismic data to the internal format with:
    - Trace data stored as Zarr arrays (lazy, chunked, compressed)
    - Headers stored as normalized Parquet files (source, receiver, trace)
    - Metadata stored as JSON
    """
    
    def __init__(self, root_path: Union[str, Path], overwrite: bool = False):
        """
        Initialize the writer.
        
        Args:
            root_path: Root directory for the output dataset.
            overwrite: If True, overwrite existing dataset. If False, raise FileExistsError if dataset exists.
        """
        root_path = Path(root_path)
        
        if root_path.exists():
            if not overwrite:
                raise FileExistsError(f"Dataset already exists at {root_path}. Set overwrite=True to replace it.")
            else:
                SeismicDatasetLayout.delete(root_path)
                
        self.layout = SeismicDatasetLayout.create(root_path)
        
    def initialize_data(
        self,
        shape: tuple,
        chunks: Optional[tuple] = None,
        dtype: Any = np.float32,
        compressor: Optional[str] = 'blosc',
        compression_level: int = 5
    ) -> None:
        """
        Initialize the Zarr array for trace data with a specific shape.
        
        Args:
            shape: (n_traces, n_samples)
            chunks: Chunk size.
            dtype: Data type.
            compressor: 'blosc', 'zstd', 'gzip', or None.
            compression_level: 1-9.
        """
        # Set up compressor
        if compressor == 'blosc':
            from numcodecs import Blosc
            comp = Blosc(cname='zstd', clevel=compression_level)
        elif compressor == 'zstd':
            from numcodecs import Zstd
            comp = Zstd(level=compression_level)
        elif compressor == 'gzip':
            from numcodecs import GZip
            comp = GZip(level=compression_level)
        elif compressor is None:
            comp = None
        else:
            raise ValueError(f"Unknown compressor: {compressor}")

        if chunks is None:
            chunks = (min(1000, shape[0]), shape[1])

        traces_group_path = str(self.layout.traces_path)
        root = zarr.open_group(traces_group_path, mode='a', zarr_version=2)
        
        # Create dataset
        root.create_dataset(
            'data',
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=comp,
            overwrite=True
        )

    def write_data_chunk(self, data: np.ndarray, start_trace: int) -> None:
        """
        Write a chunk of trace data to the initialized array.
        
        Args:
            data: Numpy array of shape (n_chunk, n_samples).
            start_trace: Index of the first trace in this chunk.
        """
        traces_group_path = str(self.layout.traces_path)
        root = zarr.open_group(traces_group_path, mode='r+', zarr_version=2)
        
        if 'data' not in root:
             raise RuntimeError("Data array not initialized. Call initialize_data() first.")
             
        z = root['data']
        end_trace = start_trace + data.shape[0]
        z[start_trace:end_trace, :] = data

    def write_traces(
        self,
        data: Union[np.ndarray, da.Array],
        chunks: Optional[tuple] = None,
        compressor: Optional[str] = 'blosc',
        compression_level: int = 5
    ) -> None:
        """
        Write entire trace data to Zarr array (Overwrites existing).
        
        Args:
            data: Trace data array (n_traces, n_samples).
            chunks: Chunk size for Zarr array. If None, uses automatic chunking.
            compressor: Compression algorithm ('blosc', 'zstd', 'gzip', or None).
            compression_level: Compression level (1-9).
        """
        if isinstance(data, da.Array):
            # Dask handles writing efficiently
            # Set up compressor
            if compressor == 'blosc':
                from numcodecs import Blosc
                comp = Blosc(cname='zstd', clevel=compression_level)
            elif compressor == 'zstd':
                 from numcodecs import Zstd
                 comp = Zstd(level=compression_level)
            elif compressor == 'gzip':
                from numcodecs import GZip
                comp = GZip(level=compression_level)
            elif compressor is None:
                comp = None
            else:
                raise ValueError(f"Unknown compressor: {compressor}")

            traces_group_path = str(self.layout.traces_path)
            data.to_zarr(
                traces_group_path,
                component='data',
                overwrite=True,
                compressor=comp,
                zarr_version=2
            )
        else:
            # For numpy, use initialize + write_chunk logic implicitly
            # to reuse code, but here we just do it in one go which is what
            # write_traces implies (writes the whole thing).
            self.initialize_data(
                shape=data.shape, 
                chunks=chunks, 
                dtype=data.dtype, 
                compressor=compressor, 
                compression_level=compression_level
            )
            self.write_data_chunk(data, 0)
    
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write global metadata to JSON file.
        
        Args:
            metadata: Dictionary containing global metadata.
        """
        # Ensure metadata directory exists (handled by layout.create/ensure_structure)
        with open(self.layout.global_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def append_provenance(self, event: Dict[str, Any]) -> None:
        """
        Append an event to the provenance history.
        
        Args:
            event: Dictionary describing the event (e.g., action, timestamp, parameters).
        """
        import yaml
        import datetime
        import getpass
        
        # Ensure base fields
        if 'timestamp' not in event:
            event['timestamp'] = datetime.datetime.utcnow().isoformat()
        if 'user' not in event:
            event['user'] = getpass.getuser()
            
        # Load existing provenance
        if self.layout.provenance_path.exists():
            with open(self.layout.provenance_path, 'r') as f:
                provenance = yaml.safe_load(f) or {'history': []}
        else:
            provenance = {'history': []}
            
        # Append new event
        if 'history' not in provenance:
            provenance['history'] = []
        provenance['history'].append(event)
        
        # Save
        with open(self.layout.provenance_path, 'w') as f:
            yaml.dump(provenance, f)

    def write_headers(
        self,
        trace_headers: pd.DataFrame,
        source_headers: Optional[pd.DataFrame] = None,
        receiver_headers: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Write headers to Parquet files.
        
        Args:
            trace_headers: DataFrame containing trace attributes.
            source_headers: Optional DataFrame containing source attributes.
            receiver_headers: Optional DataFrame containing receiver attributes.
        """
        # Validate headers against schemas (Issue #62)
        from .schema import SchemaManager
        manager = SchemaManager(self.layout.root_path)
        
        manager.validate_dataframe(trace_headers, "trace_header")
        manager.validate_dataframe(source_headers, "source")
        manager.validate_dataframe(receiver_headers, "receiver")
        
        # Write trace headers
        trace_headers.to_parquet(self.layout.trace_metadata_path)
        
        # Write source headers
        if source_headers is not None:
            source_headers.to_parquet(self.layout.source_metadata_path)
            
        # Write receiver headers
        if receiver_headers is not None:
            receiver_headers.to_parquet(self.layout.receiver_metadata_path)
    
    def write_metadata_files(
        self,
        signature: Optional[pd.DataFrame] = None,
        properties: Optional[Dict[str, Any]] = None,
        survey: Optional[Dict[str, Any]] = None,
        instrument: Optional[Dict[str, Any]] = None,
        job: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write metadata files.
        
        Args:
            signature: DataFrame containing signature data (written to Parquet).
            properties: Dictionary containing properties metadata (written to YAML).
            survey: Dictionary containing survey metadata (written to YAML).
            instrument: Dictionary containing instrument metadata (written to YAML).
            job: Dictionary containing job metadata (written to YAML).
        """
        import yaml
        
        # Write signature (Parquet)
        if signature is not None:
            signature.to_parquet(self.layout.signature_metadata_path)
            
        # Write properties (YAML)
        if properties is not None:
            with open(self.layout.properties_metadata_path, 'w') as f:
                yaml.dump(properties, f, sort_keys=False)
                
        # Write survey (YAML)
        if survey is not None:
            with open(self.layout.survey_metadata_path, 'w') as f:
                yaml.dump(survey, f, sort_keys=False)
                
        # Write instrument (YAML)
        if instrument is not None:
            with open(self.layout.instrument_metadata_path, 'w') as f:
                yaml.dump(instrument, f, sort_keys=False)
                
        # Write job (YAML)
        if job is not None:
            with open(self.layout.job_metadata_path, 'w') as f:
                yaml.dump(job, f, sort_keys=False)

