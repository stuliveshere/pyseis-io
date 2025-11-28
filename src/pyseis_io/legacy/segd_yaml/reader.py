from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pyseis_io.base import SeismicReader

from .filemap import ChannelSetInfo, GeneralHeaderInfo, SegDFileMap, TraceLayout
from .spec_loader import (
    BlockSpec,
    ChannelSetPlan,
    CountPlan,
    TraceHeaderPlan,
    VariableSectionPlan,
    load_format_spec,
)
from .types import dtype_for_sample_format, field_size, unpack_value


class SegDYamlReader(SeismicReader):
    """YAML-driven SEG-D reader that exposes a SeismicReader-compatible API."""

    def __init__(self, filename: str, profile: str) -> None:
        if not filename:
            raise ValueError("filename is required")
        if not profile:
            raise ValueError("profile must be provided explicitly")
        self._path = Path(filename)
        if not self._path.exists():
            raise FileNotFoundError(f"SEG-D file '{filename}' does not exist")
        self._spec = load_format_spec(profile)
        self._file_handle = self._path.open("rb")
        self._mmap = mmap.mmap(self._file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._buffer = memoryview(self._mmap)
        self._global_context: Dict[str, Any] = {}
        self._file_map = self._build_file_map()
        trace_plan = self._spec.layout.trace_headers
        self._demux_block = self._spec.blocks[trace_plan.demux_block]
        self._extension_blocks = [self._spec.blocks[name] for name in trace_plan.extension_blocks]

    def __enter__(self) -> "SegDYamlReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "_buffer") and self._buffer is not None:
            self._buffer.release()
            self._buffer = None  # type: ignore
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
            self._mmap = None  # type: ignore
        if hasattr(self, "_file_handle") and self._file_handle:
            self._file_handle.close()
            self._file_handle = None  # type: ignore

    def read(self) -> None:
        """Metadata is parsed eagerly during construction; nothing else to do."""
        return None

    @property
    def num_traces(self) -> int:
        return self._file_map.num_traces

    @property
    def samples_per_trace(self) -> int:
        counts = {channel.trace_layout.samples_per_trace for channel in self._file_map.channel_sets}
        if len(counts) != 1:
            raise ValueError("Channel sets declare different samples per trace")
        return counts.pop()

    @property
    def sample_rate(self) -> float:
        intervals = {channel.trace_layout.sample_interval_us for channel in self._file_map.channel_sets}
        if len(intervals) != 1:
            raise ValueError("Channel sets declare different sample intervals")
        return float(intervals.pop())

    @property
    def file_map(self) -> SegDFileMap:
        return self._file_map

    def get_trace_data(self, index: int) -> np.ndarray:
        channel = self._file_map.channel_set_for_trace(index)
        layout = channel.trace_layout
        local_index = index - channel.first_trace_index
        trace_offset = channel.first_trace_offset + local_index * layout.stride
        data_offset = trace_offset + layout.data_offset
        count = layout.samples_per_trace
        array = np.frombuffer(self._buffer, dtype=layout.dtype, count=count, offset=data_offset)
        return array.copy()

    def get_trace_header(self, index: int) -> Dict[str, Any]:
        channel = self._file_map.channel_set_for_trace(index)
        layout = channel.trace_layout
        local_index = index - channel.first_trace_index
        trace_offset = channel.first_trace_offset + local_index * layout.stride
        demux_values = self._parse_block(self._demux_block, trace_offset, self._global_context)
        extension_values: Dict[str, Dict[str, Any]] = {}
        cursor = trace_offset + self._demux_block.size
        for block in self._extension_blocks:
            extension_values[block.name] = self._parse_block(block, cursor, self._global_context)
            cursor += block.size
        return {"demux_header": demux_values, "extensions": extension_values}

    def _build_file_map(self) -> SegDFileMap:
        context: Dict[str, Any] = {}
        offset = 0
        general_header_blocks: Dict[str, List[Dict[str, Any]]] = {}
        total_general_header_bytes = 0
        for plan in self._spec.layout.general_headers:
            block = self._spec.blocks[plan.block_name]
            count = self._resolve_count(plan.count, context)
            entries: List[Dict[str, Any]] = []
            for idx in range(count):
                block_offset = offset + idx * block.size
                self._assert_within_file(block_offset, block.size)
                block_values = self._parse_block(block, block_offset, context)
                entries.append(block_values)
            general_header_blocks[block.name] = entries
            context[block.name] = entries[0] if len(entries) == 1 else entries
            offset += count * block.size
            total_general_header_bytes += count * block.size
        extended_bytes = self._compute_variable_area(self._spec.layout.extended_headers, context)
        external_bytes = self._compute_variable_area(self._spec.layout.external_headers, context)
        channel_plan = self._spec.layout.channel_sets
        channel_count = self._resolve_path(context, channel_plan.count_field)
        if not isinstance(channel_count, int) or channel_count <= 0:
            raise ValueError("channel set count must be a positive integer")
        channel_block = self._spec.blocks[channel_plan.block_name]
        channel_entries: List[Dict[str, Any]] = []
        for idx in range(channel_count):
            block_offset = offset + idx * channel_block.size
            self._assert_within_file(block_offset, channel_block.size)
            channel_entries.append(self._parse_block(channel_block, block_offset, context))
        context[channel_block.name] = channel_entries if len(channel_entries) > 1 else channel_entries[0]
        offset += channel_count * channel_block.size
        trace_data_offset = offset + extended_bytes + external_bytes
        header_bytes = bytes(self._buffer[:trace_data_offset])
        channel_infos = self._build_channel_sets(
            channel_entries,
            trace_data_offset,
            self._spec.layout.trace_headers,
            channel_plan,
        )
        general_info = GeneralHeaderInfo(
            blocks=general_header_blocks,
            total_bytes=total_general_header_bytes,
            extended_header_bytes=extended_bytes,
            external_header_bytes=external_bytes,
        )
        self._global_context = context
        return SegDFileMap(
            path=self._path,
            buffer=self._buffer,
            header_bytes=header_bytes,
            trace_data_offset=trace_data_offset,
            general_headers=general_info,
            channel_sets=tuple(channel_infos),
            data_byte_order=self._spec.layout.data_samples.byte_order,
        )

    def _build_channel_sets(
        self,
        channel_entries: Sequence[Dict[str, Any]],
        trace_start_offset: int,
        trace_plan: TraceHeaderPlan,
        channel_plan: ChannelSetPlan,
    ) -> Sequence[ChannelSetInfo]:
        demux_block = self._spec.blocks[trace_plan.demux_block]
        extension_blocks = [self._spec.blocks[name] for name in trace_plan.extension_blocks]
        extension_sizes = {block.name: block.size for block in extension_blocks}
        channel_infos: List[ChannelSetInfo] = []
        current_trace_index = 0
        current_trace_offset = trace_start_offset
        for idx, header in enumerate(channel_entries):
            trace_count = self._require_int(header, channel_plan.trace_count_field)
            samples_per_trace = self._require_int(header, channel_plan.samples_field)
            sample_interval = self._require_int(header, channel_plan.sample_interval_field)
            format_code = self._require_int(header, channel_plan.format_field)
            demux_bytes = self._require_int(header, channel_plan.trace_header_bytes_field)
            extension_bytes = self._require_int(header, channel_plan.trace_extension_bytes_field)
            dtype = dtype_for_sample_format(format_code, self._spec.layout.data_samples.byte_order)
            data_bytes = samples_per_trace * dtype.itemsize
            expected_header_bytes = demux_block.size
            if demux_bytes != expected_header_bytes:
                raise ValueError(
                    f"Channel set {idx} declares demux header {demux_bytes} "
                    f"bytes but spec block '{demux_block.name}' is {expected_header_bytes}"
                )
            expected_extension_bytes = sum(extension_sizes.values())
            if extension_bytes != expected_extension_bytes:
                raise ValueError(
                    f"Channel set {idx} declares extension size {extension_bytes} "
                    f"bytes but spec defines {expected_extension_bytes}"
                )
            stride = demux_bytes + extension_bytes + data_bytes
            layout = TraceLayout(
                demux_block_name=demux_block.name,
                demux_size=demux_bytes,
                extension_blocks=tuple(block.name for block in extension_blocks),
                extension_sizes=extension_sizes,
                data_offset=demux_bytes + extension_bytes,
                data_size=data_bytes,
                samples_per_trace=samples_per_trace,
                dtype=dtype,
                sample_interval_us=sample_interval,
                stride=stride,
            )
            channel_infos.append(
                ChannelSetInfo(
                    index=idx,
                    header=header,
                    trace_count=trace_count,
                    first_trace_index=current_trace_index,
                    first_trace_offset=current_trace_offset,
                    trace_layout=layout,
                )
            )
            current_trace_index += trace_count
            current_trace_offset += trace_count * stride
        return tuple(channel_infos)

    def _require_int(self, data: Dict[str, Any], field: str) -> int:
        value = data.get(field)
        if not isinstance(value, int):
            raise ValueError(f"Field '{field}' must be an integer, got {value!r}")
        return value

    def _resolve_count(self, count_plan: CountPlan, context: Dict[str, Any]) -> int:
        if count_plan.literal is not None:
            return count_plan.literal
        value = self._resolve_path(context, count_plan.field_path)
        if not isinstance(value, int):
            raise ValueError(f"Count field '{count_plan.field_path}' must be an integer")
        return value

    def _compute_variable_area(self, plan: VariableSectionPlan, context: Dict[str, Any]) -> int:
        if plan.block_size == 0 or not plan.count_field:
            return 0
        count_value = self._resolve_path(context, plan.count_field)
        if not isinstance(count_value, int):
            raise ValueError(f"Variable section count '{plan.count_field}' must be an integer")
        return plan.block_size * count_value

    def _parse_block(self, block: BlockSpec, start_offset: int, context: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field in block.fields:
            if field.when and not _evaluate_condition(field.when, context, result):
                continue
            absolute_offset = start_offset + field.offset
            size = field_size(field.type, field.length)
            self._assert_within_file(absolute_offset, size)
            value = unpack_value(field.type, self._buffer, absolute_offset, field.length)
            if not field.name.startswith("_"):
                result[field.name] = value
        return result

    def _assert_within_file(self, offset: int, size: int) -> None:
        file_size = len(self._buffer)
        if offset < 0 or offset + size > file_size:
            raise ValueError(f"Attempt to access bytes [{offset}, {offset + size}) beyond file size {file_size}")

    def _resolve_path(self, context: Dict[str, Any], path: str) -> Any:
        tokens = _tokenize_path(path)
        if not tokens:
            raise ValueError("Empty path is not allowed")
        current: Any = context
        for token in tokens:
            if isinstance(token, int):
                if not isinstance(current, (list, tuple)):
                    raise KeyError(f"Cannot index into non-sequence while resolving '{path}'")
                current = current[token]
            else:
                if isinstance(current, list):
                    raise KeyError(f"Cannot access attribute '{token}' on list while resolving '{path}'")
                if token not in current:
                    raise KeyError(f"Path '{path}' is invalid: missing '{token}'")
                current = current[token]
        return current


def _tokenize_path(path: str) -> List[Any]:
    tokens: List[Any] = []
    for segment in path.split("."):
        tokens.extend(_split_segment(segment))
    return tokens


def _split_segment(segment: str) -> List[Any]:
    tokens: List[Any] = []
    remaining = segment
    while remaining:
        bracket_index = remaining.find("[")
        if bracket_index == -1:
            tokens.append(remaining)
            break
        if bracket_index > 0:
            tokens.append(remaining[: bracket_index])
        close_index = remaining.find("]", bracket_index)
        if close_index == -1:
            raise ValueError(f"Unmatched '[' in path segment '{segment}'")
        index_value = int(remaining[bracket_index + 1 : close_index])
        tokens.append(index_value)
        remaining = remaining[close_index + 1 :]
    return [token for token in tokens if token != ""]


def _evaluate_condition(expression: str, context: Dict[str, Any], local: Dict[str, Any]) -> bool:
    for operator in ("==", "!="):
        if operator in expression:
            left, right = expression.split(operator, 1)
            left_value = _resolve_condition_operand(left.strip(), context, local)
            right_value = _resolve_condition_operand(right.strip(), context, local)
            if operator == "==":
                return left_value == right_value
            return left_value != right_value
    raise ValueError(f"Unsupported condition expression '{expression}'")


def _resolve_condition_operand(token: str, context: Dict[str, Any], local: Dict[str, Any]) -> Any:
    literal = _try_parse_literal(token)
    if literal is not None:
        return literal
    if token in local:
        return local[token]
    return _resolve_flat_path(context, token)


def _try_parse_literal(token: str) -> Optional[Any]:
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        return int(token)
    except ValueError:
        return None


def _resolve_flat_path(context: Dict[str, Any], path: str) -> Any:
    current: Any = context
    for token in _tokenize_path(path):
        if isinstance(token, int):
            if not isinstance(current, (list, tuple)):
                raise KeyError(f"Cannot index into {type(current)} while resolving '{path}'")
            current = current[token]
        else:
            if isinstance(current, list):
                raise KeyError(f"Cannot access '{token}' from list while resolving '{path}'")
            current = current[token]
    return current