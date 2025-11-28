from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .filemap import SegDFileMap
from .spec_loader import load_format_spec


class SegDYamlWriter:
    """Minimal writer that re-emits a SEG-D file using an existing YAML layout."""

    def __init__(
        self,
        output_path: str,
        profile: str,
        file_map: SegDFileMap,
        traces: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray],
    ) -> None:
        if not output_path:
            raise ValueError("output_path is required")
        if not profile:
            raise ValueError("profile must match a YAML definition")
        self._output_path = Path(output_path)
        self._spec = load_format_spec(profile)
        self._file_map = file_map
        self._traces = self._normalize_traces(traces, file_map)

    def write(self) -> Path:
        """Write the SEG-D file and return the output path."""
        source_bytes = self._file_map.buffer.tobytes()
        mutable = bytearray(source_bytes)
        for trace_index, trace in enumerate(self._traces):
            channel = self._file_map.channel_set_for_trace(trace_index)
            layout = channel.trace_layout
            local_index = trace_index - channel.first_trace_index
            trace_offset = channel.first_trace_offset + local_index * layout.stride
            data_offset = trace_offset + layout.data_offset
            expected_samples = layout.samples_per_trace
            if trace.size != expected_samples:
                raise ValueError(
                    f"Trace {trace_index} has {trace.size} samples but layout requires {expected_samples}"
                )
            coerced = np.asarray(trace, dtype=layout.dtype)
            data = coerced.tobytes(order="C")
            if len(data) != layout.data_size:
                raise ValueError(
                    f"Trace {trace_index} serialized to {len(data)} bytes but layout expects {layout.data_size}"
                )
            mutable[data_offset : data_offset + layout.data_size] = data
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("wb") as handle:
            handle.write(mutable)
        return self._output_path

    def _normalize_traces(
        self,
        traces: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray],
        file_map: SegDFileMap,
    ) -> List[np.ndarray]:
        if isinstance(traces, np.ndarray):
            if traces.ndim != 2:
                raise ValueError("Trace array must be 2-D (num_traces x samples)")
            if traces.shape[0] != file_map.num_traces:
                raise ValueError(
                    f"Trace array declares {traces.shape[0]} traces but file map expects {file_map.num_traces}"
                )
            return [np.asarray(row) for row in traces]
        try:
            iterator = list(traces)
        except TypeError as err:
            raise TypeError("traces must be an iterable of arrays") from err
        trace_list = [np.asarray(trace) for trace in iterator]
        if len(trace_list) != file_map.num_traces:
            raise ValueError(
                f"Provided traces length {len(trace_list)} does not match file map trace count {file_map.num_traces}"
            )
        return trace_list

