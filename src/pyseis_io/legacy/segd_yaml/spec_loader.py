from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from .types import SegDYamlTypeError, field_size


class SegDYamlSpecError(ValueError):
    """Raised when a YAML SEG-D definition is invalid."""


SPEC_DIR = Path(__file__).resolve().parent / "defs"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    offset: int
    type: str
    length: Optional[int] = None
    when: Optional[str] = None


@dataclass(frozen=True)
class BlockSpec:
    name: str
    size: int
    fields: Sequence[FieldSpec]

    def field_names(self) -> List[str]:
        return [field.name for field in self.fields]


@dataclass(frozen=True)
class CountPlan:
    literal: Optional[int] = None
    field_path: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.literal is None) == (self.field_path is None):
            raise SegDYamlSpecError("CountPlan requires exactly one of literal or field_path")


@dataclass(frozen=True)
class BlockInstancePlan:
    block_name: str
    count: CountPlan


@dataclass(frozen=True)
class ChannelSetPlan:
    block_name: str
    count_field: str
    trace_count_field: str
    samples_field: str
    sample_interval_field: str
    format_field: str
    trace_header_bytes_field: str
    trace_extension_bytes_field: str


@dataclass(frozen=True)
class TraceHeaderPlan:
    demux_block: str
    extension_blocks: Sequence[str]


@dataclass(frozen=True)
class VariableSectionPlan:
    block_size: int
    count_field: Optional[str]


@dataclass(frozen=True)
class DataSamplesPlan:
    byte_order: str


@dataclass(frozen=True)
class RecordLayout:
    general_headers: Sequence[BlockInstancePlan]
    channel_sets: ChannelSetPlan
    trace_headers: TraceHeaderPlan
    extended_headers: VariableSectionPlan
    external_headers: VariableSectionPlan
    data_samples: DataSamplesPlan


@dataclass(frozen=True)
class FormatSpec:
    profile: str
    blocks: Mapping[str, BlockSpec]
    layout: RecordLayout


def _load_yaml_dict(profile: str) -> Dict[str, Any]:
    profile_path = SPEC_DIR / f"{profile}.yml"
    if not profile_path.exists():
        raise SegDYamlSpecError(f"No YAML specification found for profile '{profile}'")
    with profile_path.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise SegDYamlSpecError(f"Specification {profile} must be a mapping")
    extends = raw.get("extends")
    if extends:
        base_dict = _load_yaml_dict(extends)
        overrides = raw.get("overrides")
        if overrides is None:
            raise SegDYamlSpecError(f"Profile '{profile}' declares extends but lacks overrides section")
        merged = _merge_dicts(base_dict, overrides)
        merged["profile"] = raw.get("profile", profile)
        return merged
    return raw


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_field(block_name: str, entry: Dict[str, Any]) -> FieldSpec:
    try:
        name = entry["name"]
        offset = int(entry["offset"])
        field_type = entry["type"]
    except KeyError as err:
        raise SegDYamlSpecError(f"Field definition in block {block_name} missing {err}") from err
    length = entry.get("length")
    when = entry.get("when")
    if length is not None:
        length = int(length)
    # Validate size to ensure offsets are meaningful
    try:
        field_size(field_type, length)
    except SegDYamlTypeError as err:
        raise SegDYamlSpecError(str(err)) from err
    return FieldSpec(name=name, offset=offset, type=field_type, length=length, when=when)


def _parse_block(name: str, data: Dict[str, Any]) -> BlockSpec:
    try:
        size = int(data["size"])
        fields = data["fields"]
    except KeyError as err:
        raise SegDYamlSpecError(f"Block '{name}' must define size and fields") from err
    if size <= 0:
        raise SegDYamlSpecError(f"Block '{name}' size must be positive")
    field_specs = [_parse_field(name, field) for field in fields]
    expected_offset = 0
    for field in field_specs:
        field_span = field_size(field.type, field.length)
        if field.offset != expected_offset:
            raise SegDYamlSpecError(
                f"Block '{name}' field '{field.name}' offset {field.offset} does not match expected {expected_offset}"
            )
        expected_offset += field_span
    if expected_offset != size:
        raise SegDYamlSpecError(
            f"Block '{name}' declared size {size} but fields consume {expected_offset} bytes"
        )
    return BlockSpec(name=name, size=size, fields=field_specs)


def _parse_count_plan(data: Dict[str, Any]) -> CountPlan:
    if "count" in data:
        return CountPlan(literal=int(data["count"]))
    if "count_field" in data:
        return CountPlan(field_path=str(data["count_field"]))
    raise SegDYamlSpecError("Block entry must define either count or count_field")


def _parse_general_headers(layout: Dict[str, Any]) -> List[BlockInstancePlan]:
    entries = layout.get("general_headers")
    if not isinstance(entries, list) or not entries:
        raise SegDYamlSpecError("record_layout.general_headers must contain at least one entry")
    plans = []
    for entry in entries:
        block_name = entry.get("block")
        if not block_name:
            raise SegDYamlSpecError("Each general header entry must reference a block name")
        plans.append(BlockInstancePlan(block_name=block_name, count=_parse_count_plan(entry)))
    return plans


def _parse_channel_sets(layout: Dict[str, Any]) -> ChannelSetPlan:
    section = layout.get("channel_sets")
    if not isinstance(section, dict):
        raise SegDYamlSpecError("record_layout.channel_sets must be defined")
    required_keys = [
        "block",
        "count_field",
        "trace_count_field",
        "samples_field",
        "sample_interval_field",
        "format_field",
        "trace_header_bytes_field",
        "trace_extension_bytes_field",
    ]
    missing = [key for key in required_keys if key not in section]
    if missing:
        raise SegDYamlSpecError(f"record_layout.channel_sets missing keys: {', '.join(missing)}")
    return ChannelSetPlan(
        block_name=section["block"],
        count_field=section["count_field"],
        trace_count_field=section["trace_count_field"],
        samples_field=section["samples_field"],
        sample_interval_field=section["sample_interval_field"],
        format_field=section["format_field"],
        trace_header_bytes_field=section["trace_header_bytes_field"],
        trace_extension_bytes_field=section["trace_extension_bytes_field"],
    )


def _parse_trace_headers(layout: Dict[str, Any]) -> TraceHeaderPlan:
    section = layout.get("trace_headers")
    if not isinstance(section, dict):
        raise SegDYamlSpecError("record_layout.trace_headers must be defined")
    demux_block = section.get("demux_block")
    if not demux_block:
        raise SegDYamlSpecError("record_layout.trace_headers requires demux_block")
    extension_blocks = section.get("extension_blocks", [])
    if not isinstance(extension_blocks, list):
        raise SegDYamlSpecError("record_layout.trace_headers.extension_blocks must be a list")
    return TraceHeaderPlan(demux_block=demux_block, extension_blocks=tuple(extension_blocks))


def _parse_variable_section(section_name: str, layout: Dict[str, Any]) -> VariableSectionPlan:
    section = layout.get(section_name)
    if not isinstance(section, dict):
        raise SegDYamlSpecError(f"record_layout.{section_name} must be defined")
    block_size = int(section.get("block_size", 0))
    count_field = section.get("count_field")
    if block_size < 0:
        raise SegDYamlSpecError(f"{section_name} block_size cannot be negative")
    return VariableSectionPlan(block_size=block_size, count_field=count_field)


def _parse_data_samples(layout: Dict[str, Any]) -> DataSamplesPlan:
    section = layout.get("data_samples")
    if not isinstance(section, dict):
        raise SegDYamlSpecError("record_layout.data_samples must be defined")
    byte_order = section.get("byte_order")
    if byte_order not in {"big", "little"}:
        raise SegDYamlSpecError("record_layout.data_samples.byte_order must be 'big' or 'little'")
    return DataSamplesPlan(byte_order=byte_order)


def load_format_spec(profile: str) -> FormatSpec:
    """Load and validate a SEG-D YAML specification."""
    raw = _load_yaml_dict(profile)
    profile_name = raw.get("profile", profile)
    blocks_section = raw.get("blocks")
    if not isinstance(blocks_section, dict):
        raise SegDYamlSpecError("Specification must define a 'blocks' mapping")
    blocks = {}
    for name, block_data in blocks_section.items():
        blocks[name] = _parse_block(name, block_data)
    layout_section = raw.get("record_layout")
    if not isinstance(layout_section, dict):
        raise SegDYamlSpecError("Specification must define record_layout")
    layout = RecordLayout(
        general_headers=_parse_general_headers(layout_section),
        channel_sets=_parse_channel_sets(layout_section),
        trace_headers=_parse_trace_headers(layout_section),
        extended_headers=_parse_variable_section("extended_headers", layout_section),
        external_headers=_parse_variable_section("external_headers", layout_section),
        data_samples=_parse_data_samples(layout_section),
    )
    _validate_block_references(blocks, layout)
    return FormatSpec(profile=profile_name, blocks=blocks, layout=layout)


def _validate_block_references(blocks: Mapping[str, BlockSpec], layout: RecordLayout) -> None:
    referenced = {plan.block_name for plan in layout.general_headers}
    referenced.add(layout.channel_sets.block_name)
    referenced.add(layout.trace_headers.demux_block)
    referenced.update(layout.trace_headers.extension_blocks)
    missing = [name for name in referenced if name not in blocks]
    if missing:
        raise SegDYamlSpecError(f"Record layout references undefined blocks: {', '.join(sorted(missing))}")
    channel_block = blocks[layout.channel_sets.block_name]
    for field_name in [
        layout.channel_sets.trace_count_field,
        layout.channel_sets.samples_field,
        layout.channel_sets.sample_interval_field,
        layout.channel_sets.format_field,
        layout.channel_sets.trace_header_bytes_field,
        layout.channel_sets.trace_extension_bytes_field,
    ]:
        if field_name not in channel_block.field_names():
            raise SegDYamlSpecError(
                f"Channel set field '{field_name}' not found in block '{channel_block.name}'"
            )

