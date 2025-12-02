"""
Reader for internal seismic data format.
"""

from pathlib import Path
from typing import Union
import dask.array as da
import json

from .layout import SeismicDatasetLayout
from .dataset import SeismicData, ParquetHeaderStore

class InternalFormatReader:
    """
    Reader for seismic data in the internal Zarr/Parquet format.
    """
    
    def __init__(self, path: Union[str, Path]):
        self.layout = SeismicDatasetLayout(path)
        self.layout.ensure_structure()
        
    def read(self) -> SeismicData:
        """
        Read the dataset and return a SeismicData object.
        
        Returns:
            SeismicData: The loaded dataset.
        """
        # Read traces (lazy)
        traces_path = str(self.layout.traces_path)
        # Data must be in 'data' component
        data = da.from_zarr(traces_path, component='data')
            
        # Read headers (lazy)
        # Currently mapping trace.parquet to the main header store
        header_store = ParquetHeaderStore(str(self.layout.trace_metadata_path))
        
        # Read metadata for sample rate
        if not self.layout.global_metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.layout.global_metadata_path}")
            
        with open(self.layout.global_metadata_path, 'r') as f:
            meta = json.load(f)
            sample_rate = meta.get('sample_rate')
            
        if sample_rate is None:
            raise ValueError("sample_rate is required in metadata.json but was not found")
                
        return SeismicData(
            data=data,
            header_store=header_store,
            sample_rate=sample_rate,
            file_path=str(self.layout.root_path)
        )
