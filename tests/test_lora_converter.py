"""Conversion checks using tiny safetensors files, without loading a base model."""

import json
import struct
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_lora_to_gguf import convert, map_module_name, read_alpha


def write_adapter(path, tensors):
    header, data = {}, bytearray()
    for name, values in tensors.items():
        array = np.asarray(values, dtype=np.float32)
        start = len(data)
        data.extend(array.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [start, len(data)],
        }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def test_alpha_uses_one_adjacent_config(tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"lora": {"rank": 16, "alpha": 32}})
    )
    assert read_alpha(adapter, None) == 32
    assert read_alpha(adapter, 16) == 16


def test_alpha_does_not_search_old_run_configs(tmp_path):
    (tmp_path / "run-config.json").write_text(json.dumps({"alpha": 32}))
    with pytest.raises(FileNotFoundError):
        read_alpha(tmp_path / "adapter.safetensors", None)


def test_convert_preserves_factors_and_training_scale(tmp_path):
    adapter, output = tmp_path / "adapter.safetensors", tmp_path / "adapter.gguf"
    prefix = "backbone_model.layers.0.self_attn.q_proj"
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    b = np.arange(6, dtype=np.float32).reshape(3, 2)
    write_adapter(adapter, {prefix + ".lora_A": a, prefix + ".lora_B": b})
    assert convert(adapter, output, 4, "f16", False) == (1, 2)
    reader = gguf.GGUFReader(str(output))
    scale = reader.get_field("breeze.lora.scale")
    assert float(scale.parts[scale.data[0]][0]) == 2
    factors = {t.name: t.data for t in reader.tensors}
    np.testing.assert_array_equal(factors["bb.blk.0.attn_q.weight.lora_a"], a)
    np.testing.assert_array_equal(factors["bb.blk.0.attn_q.weight.lora_b"], b)
    with pytest.raises(FileExistsError):
        convert(adapter, output, 4, "f16", False)


@pytest.mark.parametrize(
    "tensors",
    [
        {"lm_head.lora_A": np.ones((2, 4))},
        {"lm_head.lora_A": np.ones((2, 4)), "lm_head.lora_B": np.ones((3, 1))},
        {"lm_head.lora_A": np.ones((0, 4)), "lm_head.lora_B": np.ones((3, 0))},
        {"unknown.lora_A": np.ones((1, 4))},
    ],
)
def test_rejects_incomplete_or_unsupported_adapters(tmp_path, tensors):
    adapter, output = tmp_path / "adapter.safetensors", tmp_path / "adapter.gguf"
    write_adapter(adapter, tensors)
    with pytest.raises(ValueError):
        convert(adapter, output, 4, "f16", False)
    assert not output.exists()


def test_projection_target_mapping():
    assert map_module_name("text_encoder_proj") == "te.proj.weight"
    assert (
        map_module_name("depth_decoder.model.layers.1.mlp.down_proj")
        == "dd.blk.1.ffn_down.weight"
    )
