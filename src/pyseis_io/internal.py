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
        
    def ensure_structure(self) -> None:
        """
        Validates that the dataset exists and conforms to the layout.
        
        Raises:
            FileNotFoundError: If required files/directories are missing.
            ValueError: If layout version is incompatible.
        """
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_path}")
            
        # Check for layout metadata
        if not self.layout_metadata_path.exists():
            # Fallback for legacy datasets or partial writes could be handled here,
            # but for strict v1.0 we expect it.
            raise FileNotFoundError(f"Layout metadata not found: {self.layout_metadata_path}")
            
        # Validate version
        import yaml
        with open(self.layout_metadata_path, 'r') as f:
            meta = yaml.safe_load(f)
            
        version = meta.get('layout_version')
        if version != self.LAYOUT_VERSION:
            # Simple strict check for now. Could allow minor version compatibility later.
            raise ValueError(f"Incompatible layout version: {version}. Expected {self.LAYOUT_VERSION}")

    @classmethod
    def create(cls, path: Union[str, Path]) -> 'SeismicDatasetLayout':
        """
        Initialize a new empty dataset structure with version metadata.
        
        Args:
            path: Path to the new dataset root.
            
        Returns:
            SeismicDatasetLayout instance for the new dataset.
        """
        layout = cls(path)
        layout.root_path.mkdir(parents=True, exist_ok=True)
        layout.metadata_dir.mkdir(exist_ok=True)
        
        # Write layout metadata
        import yaml
        import datetime
        
        metadata = {
            'layout_version': cls.LAYOUT_VERSION,
            'created': datetime.datetime.utcnow().isoformat(),
            'generator': 'pyseis-io'
        }
        
        with open(layout.layout_metadata_path, 'w') as f:
            yaml.dump(metadata, f)
            
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
    
    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the writer.
        
        Args:
            root_path: Root directory for the output dataset.
        """
        root_path = Path(root_path)
        if not root_path.exists():
            self.layout = SeismicDatasetLayout.create(root_path)
        else:
            self.layout = SeismicDatasetLayout(root_path)
            self.layout.ensure_structure()
        
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
