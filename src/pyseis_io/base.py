from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class SeismicReader(ABC):
    """Abstract base class for seismic data readers."""

    @abstractmethod
    def read(self) -> None:
        """Read the entire file or prepare it for reading."""
        pass

    @property
    @abstractmethod
    def num_traces(self) -> int:
        """Return the number of traces in the dataset."""
        pass

    @property
    @abstractmethod
    def samples_per_trace(self) -> int:
        """Return the number of samples per trace."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Return the sample rate in microseconds."""
        pass

    @abstractmethod
    def get_trace_data(self, index: int) -> np.ndarray:
        """Get the data for a specific trace index."""
        pass

    @abstractmethod
    def get_trace_header(self, index: int) -> Dict[str, Any]:
        """Get the header for a specific trace index."""
        pass

class SeismicWriter(ABC):
    """Abstract base class for seismic data writers."""

    @abstractmethod
    def write(self, data: Any, headers: Any) -> None:
        """Write data and headers to a file."""
        pass
