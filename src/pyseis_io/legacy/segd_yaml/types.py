from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


class SegDYamlTypeError(ValueError):
    """Raised when a YAML-defined field type cannot be interpreted."""


_INT_TYPE_RE = re.compile(r"([ui])(\d+)(?:_(be|le))?$")
_BYTES_RE = re.compile(r"bytes(?:\[(\d+)])?$")
_BCD_RE = re.compile(r"bcd(\d+)")


def _default_byteorder(bit_count: int) -> str:
    """Return the default byte order for integral field types."""
    if bit_count <= 8:
        return "big"
    return "big"


def _parse_int_type(type_name: str) -> Optional[Dict[str, Any]]:
    match = _INT_TYPE_RE.fullmatch(type_name)
    if not match:
        return None
    signed = match.group(1) == "i"
    bit_count = int(match.group(2))
    if bit_count % 8 != 0:
        raise SegDYamlTypeError(f"Bit count {bit_count} for {type_name} is not byte aligned")
    explicit = match.group(3)
    if explicit is None:
        byteorder = _default_byteorder(bit_count)
    elif explicit == "be":
        byteorder = "big"
    elif explicit == "le":
        byteorder = "little"
    else:
        byteorder = explicit
    if byteorder not in {"big", "little"}:
        raise SegDYamlTypeError(f"Unsupported byte order {byteorder} for {type_name}")
    return {"signed": signed, "size": bit_count // 8, "byteorder": byteorder}


def _parse_bytes_type(type_name: str, explicit_length: Optional[int]) -> Optional[int]:
    match = _BYTES_RE.fullmatch(type_name)
    if not match:
        return None
    if match.group(1):
        return int(match.group(1))
    if explicit_length is None:
        raise SegDYamlTypeError(f"Type {type_name} requires an explicit length")
    return explicit_length


def _parse_bcd_type(type_name: str) -> Optional[int]:
    match = _BCD_RE.fullmatch(type_name)
    if not match:
        return None
    digits = int(match.group(1))
    if digits <= 0:
        raise SegDYamlTypeError(f"BCD digit count must be positive for {type_name}")
    return digits


def field_size(type_name: str, length: Optional[int] = None) -> int:
    """Return the number of bytes consumed by a field type."""
    int_details = _parse_int_type(type_name)
    if int_details:
        return int_details["size"]
    bytes_len = _parse_bytes_type(type_name, length)
    if bytes_len is not None:
        return bytes_len
    bcd_digits = _parse_bcd_type(type_name)
    if bcd_digits is not None:
        return (bcd_digits + 1) // 2
    raise SegDYamlTypeError(f"Unsupported field type {type_name}")


def _decode_bcd(raw: memoryview, offset: int, digits: int) -> int:
    size = (digits + 1) // 2
    data = bytes(raw[offset : offset + size])
    result_digits = []
    for byte in data:
        hi = (byte >> 4) & 0xF
        lo = byte & 0xF
        result_digits.append(str(hi))
        result_digits.append(str(lo))
    number_str = "".join(result_digits)[:digits]
    return int(number_str or "0")


def unpack_value(type_name: str, buffer: memoryview, offset: int, length: Optional[int] = None) -> Any:
    """Read a primitive value from a memory buffer according to the YAML type."""
    int_details = _parse_int_type(type_name)
    if int_details:
        size = int_details["size"]
        slice_end = offset + size
        chunk = bytes(buffer[offset:slice_end])
        return int.from_bytes(chunk, byteorder=int_details["byteorder"], signed=int_details["signed"])
    bytes_len = _parse_bytes_type(type_name, length)
    if bytes_len is not None:
        return bytes(buffer[offset : offset + bytes_len])
    bcd_digits = _parse_bcd_type(type_name)
    if bcd_digits is not None:
        return _decode_bcd(buffer, offset, bcd_digits)
    raise SegDYamlTypeError(f"Cannot unpack unsupported type {type_name}")


def pack_value(
    type_name: str,
    buffer: bytearray,
    offset: int,
    value: Any,
    length: Optional[int] = None,
) -> None:
    """Pack a primitive value into the provided buffer."""
    int_details = _parse_int_type(type_name)
    if int_details:
        size = int_details["size"]
        if not isinstance(value, int):
            raise SegDYamlTypeError(f"Expected int for {type_name}, got {type(value)}")
        buffer[offset : offset + size] = int(value).to_bytes(
            size, byteorder=int_details["byteorder"], signed=int_details["signed"]
        )
        return
    bytes_len = _parse_bytes_type(type_name, length)
    if bytes_len is not None:
        if not isinstance(value, (bytes, bytearray)):
            raise SegDYamlTypeError(f"Expected bytes for {type_name}, got {type(value)}")
        if len(value) != bytes_len:
            raise SegDYamlTypeError(f"Value for {type_name} must be {bytes_len} bytes, got {len(value)}")
        buffer[offset : offset + bytes_len] = value
        return
    bcd_digits = _parse_bcd_type(type_name)
    if bcd_digits is not None:
        if not isinstance(value, int):
            raise SegDYamlTypeError(f"Expected int for {type_name}, got {type(value)}")
        encoded = f"{value:0{bcd_digits}d}"
        if len(encoded) > bcd_digits:
            raise SegDYamlTypeError(f"Value {value} does not fit in {bcd_digits} BCD digits")
        bytes_needed = (bcd_digits + 1) // 2
        raw = bytearray(bytes_needed)
        digit_iter = iter(encoded)
        for idx in range(bytes_needed):
            hi = int(next(digit_iter, "0"))
            lo = int(next(digit_iter, "0"))
            raw[idx] = (hi << 4) | lo
        buffer[offset : offset + bytes_needed] = raw
        return
    raise SegDYamlTypeError(f"Cannot pack unsupported type {type_name}")


_SAMPLE_FORMAT_KIND: Dict[int, str] = {
    8008: "int8",
    8015: "int16",
    8036: "int32",
    8058: "float32",
    8064: "float64",
}


def dtype_for_sample_format(format_code: int, byte_order: str) -> np.dtype:
    """Translate a SEG-D data sample format into a NumPy dtype."""
    if byte_order not in {"big", "little"}:
        raise SegDYamlTypeError(f"Byte order must be 'big' or 'little', got {byte_order}")
    kind = _SAMPLE_FORMAT_KIND.get(format_code)
    if not kind:
        raise SegDYamlTypeError(f"Unsupported sample format code {format_code}")
    endianness = ">" if byte_order == "big" else "<"
    if kind == "int8":
        # Endianness is irrelevant for single-byte samples.
        return np.dtype("i1")
    if kind == "int16":
        return np.dtype(f"{endianness}i2")
    if kind == "int32":
        return np.dtype(f"{endianness}i4")
    if kind == "float32":
        return np.dtype(f"{endianness}f4")
    if kind == "float64":
        return np.dtype(f"{endianness}f8")
    raise SegDYamlTypeError(f"Unhandled kind {kind} for format {format_code}")

