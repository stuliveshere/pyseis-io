import struct
from pathlib import Path

import numpy as np
import pytest

from pyseis_io.legacy.segd_yaml.reader import SegDYamlReader
from pyseis_io.legacy.segd_yaml.writer import SegDYamlWriter
from pyseis_io.legacy.segd_yaml.spec_loader import load_format_spec


@pytest.fixture()
def segd_fixture(tmp_path: Path):
    """Create a minimal SEG-D file conforming to segd21.yml."""
    samples_per_trace = 4
    trace_count = 2
    general_header = bytearray(32)
    struct.pack_into(">H", general_header, 0, 1)
    struct.pack_into(">B", general_header, 2, 1)
    struct.pack_into(">B", general_header, 3, 10)
    struct.pack_into(">B", general_header, 4, 1)
    struct.pack_into(">B", general_header, 5, 1)
    struct.pack_into(">H", general_header, 6, samples_per_trace)
    struct.pack_into(">H", general_header, 8, 2000)
    struct.pack_into(">H", general_header, 10, 0)
    struct.pack_into(">H", general_header, 12, 0)
    struct.pack_into(">H", general_header, 14, 16)
    struct.pack_into(">H", general_header, 16, 0)
    struct.pack_into(">H", general_header, 18, 0)

    channel_header = bytearray(24)
    struct.pack_into(">B", channel_header, 0, 1)
    struct.pack_into(">B", channel_header, 1, trace_count)
    struct.pack_into(">H", channel_header, 2, samples_per_trace)
    struct.pack_into(">H", channel_header, 4, 2000)
    struct.pack_into(">H", channel_header, 6, 8058)
    struct.pack_into(">H", channel_header, 8, 16)
    struct.pack_into(">H", channel_header, 10, 0)
    struct.pack_into(">I", channel_header, 12, 1)
    struct.pack_into(">I", channel_header, 16, trace_count)

    def build_demux(sequence: int, trace_in_set: int) -> bytes:
        buf = bytearray(16)
        struct.pack_into(">I", buf, 0, sequence)
        struct.pack_into(">H", buf, 4, 1)
        struct.pack_into(">H", buf, 6, trace_in_set)
        struct.pack_into(">H", buf, 8, samples_per_trace)
        struct.pack_into(">H", buf, 10, 0)
        buf[12:16] = b"\x00" * 4
        return bytes(buf)

    traces = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    ]

    content = bytearray()
    content.extend(general_header)
    content.extend(channel_header)
    for idx, samples in enumerate(traces, start=1):
        content.extend(build_demux(idx, idx))
        content.extend(samples.astype(">f4").tobytes())

    segd_path = tmp_path / "fixture.segd"
    segd_path.write_bytes(content)
    return segd_path, traces


def test_spec_loader_profiles():
    segd_spec = load_format_spec("segd21")
    assert segd_spec.profile == "segd21"
    smart_spec = load_format_spec("smartsolo10")
    assert smart_spec.layout.data_samples.byte_order == "little"


def test_reader_parses_mock_file(segd_fixture):
    path, traces = segd_fixture
    reader = SegDYamlReader(str(path), profile="segd21")
    assert reader.num_traces == 2
    np.testing.assert_allclose(reader.get_trace_data(0), traces[0])
    header = reader.get_trace_header(0)
    assert header["demux_header"]["trace_sequence"] == 1
    reader.close()


def test_reader_out_of_bounds(segd_fixture):
    path, _ = segd_fixture
    reader = SegDYamlReader(str(path), profile="segd21")
    with pytest.raises(IndexError):
        reader.get_trace_data(10)
    reader.close()


def test_writer_round_trip(segd_fixture, tmp_path):
    source_path, _ = segd_fixture
    reader = SegDYamlReader(str(source_path), profile="segd21")
    new_traces = []
    for idx in range(reader.num_traces):
        samples = np.arange(reader.samples_per_trace, dtype=np.float32) + idx
        new_traces.append(samples)
    output_path = tmp_path / "roundtrip.segd"
    writer = SegDYamlWriter(str(output_path), "segd21", reader.file_map, new_traces)
    writer.write()
    new_reader = SegDYamlReader(str(output_path), profile="segd21")
    for idx, expected in enumerate(new_traces):
        np.testing.assert_allclose(new_reader.get_trace_data(idx), expected)
    new_reader.close()
    reader.close()

