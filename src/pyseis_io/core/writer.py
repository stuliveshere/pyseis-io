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
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write global metadata to JSON file. Merges with existing metadata.
        
        Args:
            metadata: Dictionary containing global metadata.
        """
        import json
        
        current_meta = {}
        if self.layout.global_metadata_path.exists():
            try:
                with open(self.layout.global_metadata_path, 'r') as f:
                    current_meta = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        current_meta.update(metadata)
        
        with open(self.layout.global_metadata_path, 'w') as f:
            json.dump(current_meta, f, indent=2)

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

    def write_headers(self, trace_headers: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Write headers with strict normalization to Source, Receiver, and Trace tables.
        
        Args:
            trace_headers: Flat DataFrame containing all headers.
            mapping: Optional dictionary to rename columns (Format Key -> Core Key) before processing.
            
        Raises:
            ValueError: If source_id or receiver_id validation fails.
        """
        # Ensure SchemaManager is available
        from .schema import SchemaManager
        schema_mgr = SchemaManager(self.layout.root_path)
        
        # Work on a copy to allow modification (dropping globals/columns)
        df = trace_headers.copy()
        
        # Apply Mapping if provided
        if mapping:
            df = df.rename(columns=mapping)
        
        # 0. Extract Globals
        self._extract_and_write_globals(df)
        
        # 1. Normalize and Write Source Table
        self._normalize_and_write_table(
            df=df,
            schema_name='source',
            id_col='source_id',
            schema_mgr=schema_mgr
        )
        
        # 2. Normalize and Write Receiver Table
        self._normalize_and_write_table(
            df=df,
            schema_name='receiver',
            id_col='receiver_id',
            schema_mgr=schema_mgr
        )
        
        # 3. Write Remaining Trace Headers
        self._write_trace_headers(df, schema_mgr)

    def _extract_and_write_globals(self, df: pd.DataFrame) -> None:
        """
        Identify constant columns (from a specific list) and move them to metadata.json.
        Removes them from the DataFrame to prevent duplication.
        """
        global_candidates = ['num_samples', 'sample_rate', 'coordinate_scalar', 'elevation_scalar']
        metadata_updates = {}
        
        for col in global_candidates:
            if col in df.columns:
                # Check uniqueness (ignoring NaN)
                uniques = df[col].dropna().unique()
                if len(uniques) == 1:
                    # It's a global
                    val = uniques[0]
                    # Convert numpy scalar to Python native for JSON serialization
                    if hasattr(val, 'item'):
                        val = val.item()
                    metadata_updates[col] = val
                    
                    # Remove from df so it doesn't go into trace/source/receiver parquet
                    del df[col]
                    
        if metadata_updates:
            # Load existing metadata to preserve other fields
            current_meta = {}
            if self.layout.global_metadata_path.exists():
                try:
                    with open(self.layout.global_metadata_path, 'r') as f:
                        current_meta = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            
            # Update and Write
            current_meta.update(metadata_updates)
            self.write_metadata(current_meta)

    def _normalize_and_write_table(
        self, 
        df: pd.DataFrame, 
        schema_name: str, 
        id_col: str, 
        schema_mgr: Any
    ) -> None:
        """
        Extract, validate, and write a normalized table (Source or Receiver).
        """
        # Load schema definition
        # SchemaManager.install() puts schemas in .seis/schema/[name]/v1.0.yaml
        
        schema_base_dir = self.layout.schema_dir
        if not schema_base_dir.exists():
            schema_mgr.install()
            
        # Get columns for this schema
        import yaml
        schema_path = schema_base_dir / schema_name / 'v1.0.yaml'
        if not schema_path.exists():
            # If still missing, implies schema template issue or manual corruption
            # We can't normalize without schema. skip?
            # User wants strictness.
            raise FileNotFoundError(f"Schema {schema_name} not found at {schema_path}.")
            
        with open(schema_path, 'r') as f:
            schema_def = yaml.safe_load(f)
            
        schema_cols = list(schema_def.get('columns', {}).keys())
        
        # Identify columns present in DataFrame
        # Intersection
        present_cols = [c for c in schema_cols if c in df.columns]
        
        # Must include ID col
        if id_col not in present_cols:
             if id_col not in df.columns:
                 # If ID is missing entirely from input...
                 if not present_cols:
                     return # Nothing to normalize (empty intersection)
                 raise ValueError(f"Input DataFrame missing required '{id_col}' for {schema_name} normalization.")
             present_cols.append(id_col)
             present_cols = list(set(present_cols))
             
        # Extract and Dedup
        table_df = df[present_cols].drop_duplicates()
        
        # Validation: Check ID uniqueness
        if not table_df[id_col].is_unique:
            duplicates = table_df[table_df.duplicated(id_col, keep=False)]
            msg = f"{schema_name.capitalize()} ID conflict detected. The following IDs have multiple attribute sets:\n"
            msg += str(duplicates.sort_values(id_col).head(10))
            raise ValueError(msg)
            
        # Write to Parquet
        if schema_name == 'source':
            out_path = self.layout.source_metadata_path
        elif schema_name == 'receiver':
            out_path = self.layout.receiver_metadata_path
        else:
            out_path = self.layout.root_path / f"{schema_name}.parquet"
            
        table_df.to_parquet(out_path, index=False)

    def _write_trace_headers(self, df: pd.DataFrame, schema_mgr: Any) -> None:
        """
        Write the remaining trace headers (preserving foreign keys).
        """
        import yaml
        # Determine columns to KEEP
        exclude_cols = set()
        for name in ['source', 'receiver']:
            schema_path = self.layout.schema_dir / name / 'v1.0.yaml'
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    s = yaml.safe_load(f)
                    exclude_cols.update(s.get('columns', {}).keys())
        
        # Keep IDs
        exclude_cols.discard('source_id')
        exclude_cols.discard('receiver_id')
        
        # Calculate trace columns
        trace_cols = [c for c in df.columns if c not in exclude_cols or c in ['source_id', 'receiver_id']]
        
        trace_df = df[trace_cols]
        
        # Write to parquet
        out_path = self.layout.trace_metadata_path
        trace_df.to_parquet(out_path, index=False)

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
