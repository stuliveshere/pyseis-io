"""Internal format (Zarr/Parquet) reader and writer for seismic datasets."""

import os
import shutil
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
import zarr
import numpy as np
import dask.array as da


class SeismicDatasetLayout:
    """
    Manages the internal directory structure for a seismic dataset.
    
    The layout consists of:
    - traces.zarr/: Zarr group containing:
        - data: Trace data array
    - source.parquet: Source attributes
    - receiver.parquet: Receiver attributes
    - trace.parquet: Trace attributes
    - metadata/: Directory containing:
        - layout.yaml: Layout version and metadata
        - metadata.json: Global metadata (legacy/user)
    """
    
    LAYOUT_VERSION = "1.0"
    
    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the layout manager.
        
        Args:
            root_path: Root directory for the dataset.
        """
        self.root_path = Path(root_path)
        
    @property
    def traces_path(self) -> Path:
        """Path to the traces Zarr group."""
        return self.root_path / "traces.zarr"
        
    @property
    def source_metadata_path(self) -> Path:
        """Path to the source metadata Parquet file."""
        return self.root_path / "source.parquet"
        
    @property
    def receiver_metadata_path(self) -> Path:
        """Path to the receiver metadata Parquet file."""
        return self.root_path / "receiver.parquet"
        
    @property
    def trace_metadata_path(self) -> Path:
        """Path to the trace metadata Parquet file."""
        return self.root_path / "trace.parquet"
        
    @property
    def metadata_dir(self) -> Path:
        """Path to the metadata directory."""
        return self.root_path / "metadata"

    @property
    def layout_metadata_path(self) -> Path:
        """Path to the layout metadata YAML file."""
        return self.metadata_dir / "layout.yaml"
        
    @property
    def global_metadata_path(self) -> Path:
        """Path to the global metadata JSON file."""
        return self.metadata_dir / "metadata.json"
        
    @property
    def provenance_path(self) -> Path:
        """Path to the provenance YAML file."""
        return self.metadata_dir / "provenance.yaml"

    @property
    def schema_dir(self) -> Path:
        """Path to the schema directory."""
        return self.root_path / "schema"

    @property
    def layout_schema_path(self) -> Path:
        """Path to the layout schema YAML file."""
        return self.schema_dir / "layout_v1.0.yaml"
        
    def ensure_structure(self) -> None:
        """
        Validates that the dataset exists and conforms to the layout.
        
        Raises:
            FileNotFoundError: If required files/directories are missing.
            ValueError: If layout version is incompatible.
        """
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_path}")
            
        # Check for layout schema (formerly metadata/layout.yaml)
        if not self.layout_schema_path.exists():
            # Fallback check for old location during migration (optional, but good practice)
            old_layout = self.metadata_dir / "layout.yaml"
            if old_layout.exists():
                 # For now, just error out as we are in strict v1.0 mode
                 pass
            raise FileNotFoundError(f"Layout schema not found: {self.layout_schema_path}")
            
        # Validate version
        import yaml
        with open(self.layout_schema_path, 'r') as f:
            meta = yaml.safe_load(f)
            
        version = meta.get('layout_version')
        if version != self.LAYOUT_VERSION:
            # Simple strict check for now. Could allow minor version compatibility later.
            raise ValueError(f"Incompatible layout version: {version}. Expected {self.LAYOUT_VERSION}")

    @classmethod
    def create(cls, path: Union[str, Path]) -> 'SeismicDatasetLayout':
        """
        Initialize a new empty dataset structure with version metadata and schemas.
        
        Args:
            path: Path to the new dataset root.
            
        Returns:
            SeismicDatasetLayout instance for the new dataset.
        """
        layout = cls(path)
        layout.root_path.mkdir(parents=True, exist_ok=True)
        layout.metadata_dir.mkdir(exist_ok=True)
        layout.schema_dir.mkdir(exist_ok=True)
        
        # Copy schema templates
        import pkg_resources
        import glob
        
        # Try to find templates in the package
        try:
            # Modern python approach (python 3.9+)
            from importlib import resources
            # Note: This assumes pyseis_io is installed as a package. 
            # For development, we might need a fallback to relative paths if not installed.
            # However, since we are in the same package, we can try to locate them relative to this file.
            
            template_dir = Path(__file__).parent / "templates" / "schemas"
            if not template_dir.exists():
                 # Fallback for installed package structure if different
                 # This is a simplification; robust resource access is complex.
                 # Assuming editable install or source checkout for now based on file structure.
                 pass
            
            for schema_file in template_dir.glob("*.yaml"):
                shutil.copy2(schema_file, layout.schema_dir)
                
        except Exception as e:
            # Fallback or error handling
            raise RuntimeError(f"Failed to copy schema templates: {e}")

        # Initialize provenance history
        import yaml
        import datetime
        import getpass
        
        timestamp = datetime.datetime.utcnow().isoformat()
        
        provenance = {
            'history': [
                {
                    'action': 'created',
                    'timestamp': timestamp,
                    'user': getpass.getuser(),
                    'software': 'pyseis-io'
                }
            ]
        }
        
        with open(layout.provenance_path, 'w') as f:
            yaml.dump(provenance, f)
            
        return layout
        
    @staticmethod
    def rename(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Rename a dataset (implemented as a directory move).
        
        Args:
            src: Source path.
            dst: Destination path.
        """
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            raise FileNotFoundError(f"Source dataset not found: {src}")
            
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
            
        shutil.move(str(src), str(dst))
        
    @staticmethod
    def copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Deep copy a dataset (essential for copy-on-write workflows).
        
        Args:
            src: Source path.
            dst: Destination path.
        """
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            raise FileNotFoundError(f"Source dataset not found: {src}")
            
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
            
        shutil.copytree(src, dst)
        
    @staticmethod
    def delete(path: Union[str, Path]) -> None:
        """
        Remove a dataset.
        
        Args:
            path: Path to the dataset to remove.
        """
        path = Path(path)
        
        if not path.exists():
            return
            
        if path.is_dir():
            shutil.rmtree(path)
        else:
            # Should be a directory, but handle file case just in case
            path.unlink()


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
        
    def write_traces(
        self,
        data: Union[np.ndarray, da.Array],
        chunks: Optional[tuple] = None,
        compressor: Optional[str] = 'blosc',
        compression_level: int = 5
    ) -> None:
        """
        Write trace data to Zarr array.
        
        Args:
            data: Trace data array (n_traces, n_samples).
            chunks: Chunk size for Zarr array. If None, uses automatic chunking.
            compressor: Compression algorithm ('blosc', 'zstd', 'gzip', or None).
            compression_level: Compression level (1-9).
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
        
        # Determine chunks
        if chunks is None:
            # Default chunking: chunk by traces (e.g., 1000 traces at a time)
            if isinstance(data, da.Array):
                chunks = data.chunksize
            else:
                chunks = (min(1000, data.shape[0]), data.shape[1])
        
        # Write to Zarr
        # Note: We now write to the 'data' group within traces.zarr
        traces_group_path = str(self.layout.traces_path)
        
        if isinstance(data, da.Array):
            # If already a dask array, save directly
            data.to_zarr(
                traces_group_path,
                component='data',
                overwrite=True,
                compressor=comp,
                zarr_version=2
            )
        else:
            # Convert numpy array to Zarr using open_array
            # We open the group first, then create the array
            root = zarr.open_group(traces_group_path, mode='a', zarr_format=2)
            z = root.create_dataset(
                'data',
                shape=data.shape,
                chunks=chunks,
                dtype=data.dtype,
                compressor=comp,
                overwrite=True
            )
            z[:] = data
    
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
