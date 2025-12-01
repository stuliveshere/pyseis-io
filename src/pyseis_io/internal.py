import os
import shutil
from pathlib import Path
from typing import Union

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
        self.root_path = Path(root_path)
        
    @property
    def traces_path(self) -> Path:
        return self.root_path / "traces.zarr"
        
    @property
    def source_metadata_path(self) -> Path:
        return self.root_path / "source.parquet"
        
    @property
    def receiver_metadata_path(self) -> Path:
        return self.root_path / "receiver.parquet"
        
    @property
    def trace_metadata_path(self) -> Path:
        return self.root_path / "trace.parquet"
        
    @property
    def global_metadata_path(self) -> Path:
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
