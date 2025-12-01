"""
Layout management for seismic datasets.
"""

import shutil
from pathlib import Path
from typing import Union

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
    def signature_metadata_path(self) -> Path:
        """Path to the signature metadata Parquet file."""
        return self.root_path / "signature.parquet"
        
    @property
    def properties_metadata_path(self) -> Path:
        """Path to the properties metadata YAML file."""
        return self.metadata_dir / "properties.yaml"
        
    @property
    def survey_metadata_path(self) -> Path:
        """Path to the survey metadata YAML file."""
        return self.metadata_dir / "survey.yaml"
        
    @property
    def instrument_metadata_path(self) -> Path:
        """Path to the instrument metadata YAML file."""
        return self.metadata_dir / "instrument.yaml"
        
    @property
    def job_metadata_path(self) -> Path:
        """Path to the job metadata YAML file."""
        return self.metadata_dir / "job.yaml"
        
    def ensure_structure(self) -> None:
        """
        Validates that the dataset exists and conforms to the layout.
        
        Raises:
            FileNotFoundError: If required files/directories are missing.
            ValueError: If layout version is incompatible.
        """
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_path}")
            
        # Validate schemas using SchemaManager
        from .schema import SchemaManager
        manager = SchemaManager(self.root_path)
        try:
            manager.validate()
        except FileNotFoundError:
            # Fallback for datasets created before SchemaManager (Issue #42 era)
            # They might have flat schema files but no manifest.
            # For strict v1.0 compliance going forward, we could enforce it,
            # but let's check for at least the layout schema file if manifest is missing.
            layout_schema = self.schema_dir / "layout_v1.0.yaml"
            if not layout_schema.exists():
                 # Check legacy location
                 if not (self.metadata_dir / "layout.yaml").exists():
                     raise FileNotFoundError("Layout schema not found (checked manifest and legacy paths)")
        
        # We can still do a quick version check if we want, or rely on SchemaManager
        # For now, let's trust SchemaManager.validate() handles integrity.

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
        
        # Install schemas using SchemaManager
        from .schema import SchemaManager
        manager = SchemaManager(layout.root_path)
        manager.install()

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
