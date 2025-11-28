from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class TraceLayout:
    demux_block_name: str
    demux_size: int
    extension_blocks: Sequence[str]
    extension_sizes: Dict[str, int]
    data_offset: int
    data_size: int
    samples_per_trace: int
    dtype: np.dtype
    sample_interval_us: int
    stride: int


@dataclass
class ChannelSetInfo:
    index: int
    header: Dict[str, Any]
    trace_count: int
    first_trace_index: int
    first_trace_offset: int
    trace_layout: TraceLayout


@dataclass
class GeneralHeaderInfo:
    blocks: Dict[str, List[Dict[str, Any]]]
    total_bytes: int
    extended_header_bytes: int
    external_header_bytes: int

    def block(self, name: str) -> Dict[str, Any]:
        entries = self.blocks.get(name)
        if not entries:
            raise KeyError(f"General header block '{name}' not available")
        if len(entries) > 1:
            raise ValueError(f"Block '{name}' has multiple entries; specify explicitly")
        return entries[0]


@dataclass
class SegDFileMap:
    path: Path
    buffer: memoryview
    header_bytes: bytes
    trace_data_offset: int
    general_headers: GeneralHeaderInfo
    channel_sets: Sequence[ChannelSetInfo]
    data_byte_order: str

    @property
    def num_traces(self) -> int:
        return sum(channel.trace_count for channel in self.channel_sets)

    @property
    def file_size(self) -> int:
        return len(self.buffer)

    def channel_set_for_trace(self, trace_index: int) -> ChannelSetInfo:
        if trace_index < 0 or trace_index >= self.num_traces:
            raise IndexError(f"Trace index {trace_index} out of range 0..{self.num_traces - 1}")
        for channel_set in self.channel_sets:
            start = channel_set.first_trace_index
            end = start + channel_set.trace_count
            if start <= trace_index < end:
                return channel_set
        raise IndexError(f"Trace index {trace_index} not mapped to any channel set")

    def iter_channel_sets(self) -> Sequence[ChannelSetInfo]:
        return self.channel_sets

