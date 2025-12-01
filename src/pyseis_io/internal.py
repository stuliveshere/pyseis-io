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
    - traces.zarr: Zarr array containing trace data
    - source.parquet: Source attributes
    - receiver.parquet: Receiver attributes
    - trace.parquet: Trace attributes
    - metadata.json: Global metadata
    """
    
    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the layout manager.
        
        Args:
            root_path: Root directory for the dataset.
        """
        self.root_path = Path(root_path)
        
    @property
    def traces_path(self) -> Path:
        """Path to the traces Zarr array."""
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
    def global_metadata_path(self) -> Path:
        """Path to the global metadata JSON file."""
        return self.root_path / "metadata.json"
        
    def ensure_structure(self) -> None:
        """Creates the root directory if it doesn't exist."""
        self.root_path.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def create(cls, path: Union[str, Path]) -> 'SeismicDatasetLayout':
        """
        Initialize a new empty dataset structure.
        
        Args:
            path: Path to the new dataset root.
            
        Returns:
            SeismicDatasetLayout instance for the new dataset.
        """
        layout = cls(path)
        layout.ensure_structure()
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
        if isinstance(data, da.Array):
            # If already a dask array, save directly
            data.to_zarr(
                str(self.layout.traces_path),
                overwrite=True,
                compressor=comp,
                zarr_version=2  # Use v2 format for compressor compatibility
            )
        else:
            # Convert numpy array to Zarr using open_array
            z = zarr.open_array(
                str(self.layout.traces_path),
                mode='w',
                shape=data.shape,
                chunks=chunks,
                dtype=data.dtype,
                compressor=comp,
                zarr_format=2  # Use v2 format for compressor compatibility
            )
            z[:] = data
    
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write global metadata to JSON file.
        
        Args:
            metadata: Dictionary containing global metadata.
        """
        with open(self.layout.global_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
