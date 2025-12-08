"""
Core data models and structures for seismic data representation.
"""

from typing import Optional, Union, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetHeaderStore:
    """
    A lightweight wrapper around a Parquet file for efficient, lazy header access.
    
    This store is responsible for reading contiguous windows of rows from the 
    Parquet file into Arrow Tables. It does not handle complex indexing logic;
    that is delegated to the in-memory Pandas DataFrame materialized by SeismicData.
    """
    def __init__(self, path: str):
        self.path = path
        self.pq_file = pq.ParquetFile(path)
        self.num_rows = self.pq_file.metadata.num_rows
        
        # Build row group index for fast slicing
        self.row_groups = []
        start = 0
        for i in range(self.pq_file.num_row_groups):
            n = self.pq_file.metadata.row_group(i).num_rows
            self.row_groups.append({
                'id': i,
                'start': start,
                'end': start + n,
                'num_rows': n
            })
            start += n

    def read_window(self, start: int, stop: int, columns: Optional[List[str]] = None) -> pa.Table:
        """
        Read a contiguous window of rows [start:stop) as an Arrow Table.
        """
        if start < 0: start += self.num_rows
        if stop < 0: stop += self.num_rows
        start = max(0, start)
        stop = min(self.num_rows, stop)
        
        if start >= stop:
            return self.pq_file.schema.empty_table()

        # Find relevant row groups
        groups_to_read = []
        for rg in self.row_groups:
            # Check overlap: rg_start < stop AND rg_end > start
            if rg['start'] < stop and rg['end'] > start:
                groups_to_read.append(rg['id'])
        
        if not groups_to_read:
            return self.pq_file.schema.empty_table()

        # Read the row groups
        table = self.pq_file.read_row_groups(groups_to_read, columns=columns)
        
        # Calculate offset within the read table
        first_group_start = self.row_groups[groups_to_read[0]]['start']
        rel_start = start - first_group_start
        length = stop - start
        
        return table.slice(rel_start, length)
    
    def __len__(self):
        return self.num_rows

    def close(self):
        """Release the ParquetFile resource."""
        self.pq_file = None

class SeismicData:
    """
    A lazy-loading container for seismic data and headers.
    
    Attributes:
        data (dask.array.Array): Lazy trace data (traces x samples).
        header_store (ParquetHeaderStore): The source of truth for headers.
        sample_rate (float): Sample rate in microseconds.
    """

    def __init__(
        self,
        data: da.Array,
        header_store: ParquetHeaderStore,
        sample_rate: float,
        file_path: Optional[str] = None,
        _trace_slice: Optional[slice] = None
    ):
        self.data = data
        self.header_store = header_store
        self.sample_rate = sample_rate
        self.file_path = file_path
        
        # Internal state to track the current view window
        # If None, represents the full dataset
        self._trace_slice = _trace_slice or slice(0, data.shape[0], 1)

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]
        
    @property
    def headers(self) -> pd.DataFrame:
        """
        Materialize the headers for the current view as an Arrow-backed Pandas DataFrame.
        """
        start, stop, step = self._trace_slice.indices(len(self.header_store))
        
        # Read the window from Parquet
        # Note: Parquet reading doesn't support step, so we read the range and then slice in memory if needed
        table = self.header_store.read_window(start, stop)
        
        # Convert to Pandas with Arrow backend for efficiency
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        
        # Apply step if needed
        if step != 1:
            df = df.iloc[::step]
            
        return df

    def __getitem__(self, key: Union[int, slice]) -> 'SeismicData':
        """
        Slice the dataset (lazy).
        
        Returns a new SeismicData instance representing the view.
        """
        # Slice data (lazy)
        new_data = self.data[key]
        
        # Update trace slice
        # We need to compose the new slice 'key' with the existing '_trace_slice'
        
        current_start, current_stop, current_step = self._trace_slice.indices(len(self.header_store))
        
        if isinstance(key, slice):
            # Normalize key relative to the current view size
            view_len = (current_stop - current_start + current_step - 1) // current_step
            k_start, k_stop, k_step = key.indices(view_len)
            
            # Map back to absolute coordinates
            new_start = current_start + k_start * current_step
            new_stop = current_start + k_stop * current_step
            new_step = current_step * k_step
            
            # Clamp stop to ensure we don't go beyond original bounds logic (indices handles this mostly)
            # But we need to be careful about the stop condition in range logic vs slice logic
            
            new_slice = slice(new_start, new_stop, new_step)
            
        elif isinstance(key, int):
            # Single row view
            # Normalize key
            view_len = (current_stop - current_start + current_step - 1) // current_step
            if key < 0: key += view_len
            if key < 0 or key >= view_len:
                raise IndexError("Trace index out of range")
            
            abs_idx = current_start + key * current_step
            new_slice = slice(abs_idx, abs_idx + 1, 1)
            
        else:
            raise TypeError("Invalid slice key")

        return SeismicData(
            new_data, 
            self.header_store, 
            self.sample_rate, 
            self.file_path, 
            _trace_slice=new_slice
        )

    def compute(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Trigger computation and return in-memory objects.
        """
        return self.data.compute(), self.headers

    def close(self):
        """
        Release resources.
        """
        if self.header_store:
            self.header_store.close()

    @classmethod
    def open(cls, path: Union[str, Path]) -> 'SeismicData':
        """
        Open a seismic dataset from disk.
        
        Args:
            path: Path to the dataset.
            
        Returns:
            SeismicData: The loaded dataset.
        """
        from .reader import InternalFormatReader
        reader = InternalFormatReader(path)
        return reader.read()
        
    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Save the seismic dataset to disk.
        
        Args:
            path: Destination path.
            overwrite: Whether to overwrite existing dataset.
        """
        from .writer import InternalFormatWriter
        writer = InternalFormatWriter(path, overwrite=overwrite)
        
        # Write traces
        writer.write_traces(self.data)
        
        # Write headers
        # Currently writing all headers to trace_headers
        # TODO: Support normalized header writing if self.headers is structured that way
        writer.write_headers(trace_headers=self.headers)
        
        # Write metadata
        writer.write_metadata({'sample_rate': self.sample_rate})

    def summary(self) -> str:
        """
        Return a textual summary of the dataset (dimensions, geometry, key ranges).
        Similar to surange.
        """
        lines = []
        lines.append(f"SeismicData Summary:")
        lines.append(f"-------------------")
        lines.append(f"Source: {self.file_path or 'Memory'}")
        lines.append(f"Traces: {self.n_traces}")
        lines.append(f"Samples: {self.n_samples}")
        
        # sample_rate is in seconds
        us = self.sample_rate * 1_000_000.0
        ms = self.sample_rate * 1_000.0
        lines.append(f"Sample Rate: {us:.2f} us ({ms:.2f} ms)")
        
        duration = self.n_samples * self.sample_rate # seconds
        lines.append(f"Length: {duration:.2f} s")
        
        # approximate size
        size_bytes = self.n_traces * self.n_samples * 4 # float32
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                lines.append(f"Raw Data Size: {size_bytes:.2f} {unit}")
                break
            size_bytes /= 1024.0
            
        lines.append("")
        lines.append("Header Statistics:")
        
        # We need to read specific columns to get stats efficiently
        # If dataset is huge, this might take a moment, but Parquet is fast.
        # Key columns to check
        keys = ['trace_id', 'source_id', 'source_index', 'fldr', 'cdp_id', 
                'source_x', 'source_y', 'source_z',
                'receiver_x', 'receiver_y', 'receiver_z',
                'offset']
        
        # Available columns in parquet
        # self.header_store.pq_file.schema.names
        avail_cols = [c for c in keys if c in self.header_store.pq_file.schema.names]
        
        if not avail_cols:
             lines.append("  No standard key headers found.")
        else:
             # Read only these columns
             # Using header_store read_window for full range
             # Ideally we'd use pyarrow's statistics if available, but simplest is to read columns 
             # (parquet is columnar, so this is cheap-ish)
             table = self.header_store.read_window(0, self.n_traces, columns=avail_cols)
             df = table.to_pandas()
             
             for col in avail_cols:
                 series = df[col]
                 # Try to convert to numeric if possible for min/max
                 try:
                     # Check if it looks numeric
                     numeric = pd.to_numeric(series, errors='ignore')
                     min_val = numeric.min()
                     max_val = numeric.max()
                     lines.append(f"  {col:<15}: {min_val} to {max_val}")
                 except:
                     lines.append(f"  {col:<15}: (non-numeric)")

        return "\n".join(lines)

