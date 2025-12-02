"""
Schema management for pyseis-io datasets.

Handles schema template access, installation, versioning, and validation.
"""

import hashlib
import yaml
import datetime
from pathlib import Path
from typing import Dict, Any
import importlib.resources

class SchemaManager:
    """
    Manages schema lifecycle for a dataset.
    """
    
    MANIFEST_FILENAME = "schema_manifest.yaml"
    SCHEMA_DIR_NAME = "schema"
    
    def __init__(self, dataset_root: Path):
        """
        Initialize the schema manager.
        
        Args:
            dataset_root: Root directory of the dataset.
        """
        self.dataset_root = Path(dataset_root)
        self.schema_dir = self.dataset_root / self.SCHEMA_DIR_NAME
        self.manifest_path = self.dataset_root / "metadata" / self.MANIFEST_FILENAME

    def install(self) -> None:
        """
        Install schema templates to the dataset.
        
        Creates the versioned directory structure and generates the manifest.
        """
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        installed_schemas = {}
        
        # Access templates using importlib.resources
        # We assume templates are in pyseis_io.templates.schemas
        # Note: This requires the templates directory to be a python package (have __init__.py)
        # or use files() API for python 3.9+
        
        try:
            # Python 3.9+ API
            template_files = importlib.resources.files('pyseis_io.templates.schemas').glob('*.yaml')
        except (ImportError, AttributeError):
             # Fallback for older python or if structure differs slightly
             # For now assuming 3.9+ as per pyproject.toml requires-python
             raise RuntimeError("Python 3.9+ required for schema resources")

        for template_path in template_files:
            if not template_path.is_file():
                continue
                
            filename = template_path.name
            # Parse filename to determine component and version
            # Expected format: [component]_v[version].yaml
            # e.g., source_v1.0.yaml, layout_v1.0.yaml
            
            if "_v" not in filename:
                continue
                
            component, version_ext = filename.split("_v", 1)
            version = version_ext.replace(".yaml", "")
            
            # Create component directory
            # e.g., schema/source/
            component_dir = self.schema_dir / component
            component_dir.mkdir(exist_ok=True)
            
            # Target path: schema/source/v1.0.yaml
            target_filename = f"v{version}.yaml"
            target_path = component_dir / target_filename
            
            # Copy file content
            content = template_path.read_bytes()
            with open(target_path, 'wb') as f:
                f.write(content)
                
            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()
            
            installed_schemas[component] = {
                "path": f"{self.SCHEMA_DIR_NAME}/{component}/{target_filename}",
                "version": version,
                "checksum": f"sha256:{checksum}"
            }
            
        self._write_manifest(installed_schemas)

    def _write_manifest(self, schemas: Dict[str, Any]) -> None:
        """Write the schema manifest file."""
        # Get version using importlib.metadata
        try:
            from importlib.metadata import version
            pyseis_io_version = version("pyseis-io")
        except Exception:
            # Fallback if not installed or in development
            pyseis_io_version = "unknown"
        
        manifest = {
            "schemas": schemas,
            "generated_by": {
                "pyseis_io_version": pyseis_io_version,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        }
        
        # Ensure metadata dir exists
        self.manifest_path.parent.mkdir(exist_ok=True)
        
        with open(self.manifest_path, 'w') as f:
            yaml.dump(manifest, f, sort_keys=False)

    def validate(self) -> None:
        """
        Validate the dataset schemas against the manifest.
        
        Raises:
            FileNotFoundError: If manifest or schema files are missing.
            ValueError: If checksums do not match (integrity check failure).
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Schema manifest not found: {self.manifest_path}")
            
        with open(self.manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
            
        schemas = manifest.get("schemas", {})
        
        for component, info in schemas.items():
            rel_path = info.get("path")
            expected_checksum = info.get("checksum")
            
            if not rel_path:
                continue
                
            full_path = self.dataset_root / rel_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"Schema file missing for {component}: {full_path}")
                
            # Verify checksum
            if expected_checksum:
                algo, hash_val = expected_checksum.split(":", 1)
                if algo != "sha256":
                    # Warn or skip unknown algos
                    continue
                    
                with open(full_path, 'rb') as f:
                    content = f.read()
                    actual_hash = hashlib.sha256(content).hexdigest()
                    
                if actual_hash != hash_val:
                    raise ValueError(
                        f"Schema integrity check failed for {component}. "
                        f"Expected {hash_val}, got {actual_hash}"
                    )
