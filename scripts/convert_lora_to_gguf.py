"""把训练得到的 Breeze LoRA 转成可由 C++ 服务加载的 GGUF。

Usage:
    python scripts/convert_lora_to_gguf.py adapter.safetensors -o adapter.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np
from convert_hf_to_gguf import Safetensors

FORMAT = "breeze-lora-v1"
LAYER_TARGETS = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
PROJECTION_TARGETS = {
    "text_encoder_proj": "te.proj.weight",
    "depth_decoder.model.inputs_embeds_projector": "dd.in_proj.weight",
    "lm_head": "bb.lm_head.weight",
}


def map_module_name(name: str) -> str:
    """将训练层的名称转换为底模中对应的 GGUF 权重名称。"""
    if name in PROJECTION_TARGETS:
        return PROJECTION_TARGETS[name]

    for source_prefix, target_prefix in (
        ("backbone_model.layers.", "bb"),
        ("depth_decoder.model.layers.", "dd"),
    ):
        if not name.startswith(source_prefix):
            continue
        layer, separator, projection = name[len(source_prefix) :].partition(".")
        if not separator or not layer.isdigit() or projection not in LAYER_TARGETS:
            break
        return f"{target_prefix}.blk.{layer}.{LAYER_TARGETS[projection]}.weight"

    raise ValueError(f"unsupported Breeze LoRA target: {name}")


def read_alpha(adapter_path: Path, explicit_alpha: float | None) -> float:
    """读取与权重相邻的唯一配置；命令行 alpha 可明确覆盖配置值。"""
    if explicit_alpha is not None:
        alpha = explicit_alpha
    else:
        config_path = adapter_path.with_name("adapter_config.json")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        alpha = data["lora"]["alpha"]
    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be a positive finite number")
    return float(alpha)


def convert(
    adapter_path: Path, output_path: Path, alpha: float, dtype: str, force: bool
) -> tuple[int, float]:
    """检查 A/B 权重是否配对，再写入 LoRA 权重及 alpha/rank 缩放值。"""
    reader = Safetensors(adapter_path)
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for key in reader.names():
        if key.endswith(".lora_A"):
            module_name, factor = key[: -len(".lora_A")], "lora_a"
        elif key.endswith(".lora_B"):
            module_name, factor = key[: -len(".lora_B")], "lora_b"
        else:
            raise ValueError(f"unexpected tensor in adapter: {key}")
        base_name = map_module_name(module_name)
        factors = pairs.setdefault(base_name, {})
        if factor in factors:
            raise ValueError(f"duplicate {factor} for {base_name}")
        factors[factor] = reader.get(key)

    if not pairs:
        raise ValueError("adapter contains no LoRA factors")

    ranks: set[int] = set()
    for base_name, factors in pairs.items():
        if set(factors) != {"lora_a", "lora_b"}:
            raise ValueError(f"adapter is missing a paired factor for {base_name}")
        a, b = factors["lora_a"], factors["lora_b"]
        if (
            a.ndim != 2
            or b.ndim != 2
            or min(a.shape + b.shape) < 1
            or a.shape[0] != b.shape[1]
        ):
            raise ValueError(
                f"invalid factor shapes for {base_name}: A={a.shape}, B={b.shape}"
            )
        ranks.add(int(a.shape[0]))
    if len(ranks) != 1:
        raise ValueError(f"adapter layers use inconsistent ranks: {sorted(ranks)}")
    rank = ranks.pop()
    scale = alpha / rank

    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists (pass --force to replace): {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    temp_path.unlink(missing_ok=True)

    writer = gguf.GGUFWriter(str(temp_path), "breeze-lora")
    writer.add_string("breeze.lora.format", FORMAT)
    writer.add_float32("breeze.lora.scale", scale)
    out_dtype = np.float16 if dtype == "f16" else np.float32
    try:
        for base_name, factors in pairs.items():
            for factor_name, array in factors.items():
                writer.add_tensor(
                    f"{base_name}.{factor_name}",
                    np.ascontiguousarray(array, dtype=out_dtype),
                )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        temp_path.replace(output_path)
    except Exception:
        writer.close()
        temp_path.unlink(missing_ok=True)
        raise

    return len(pairs), scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path, help="input adapter.safetensors file")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output adapter GGUF path"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="training alpha (read from adjacent adapter_config.json when omitted)",
    )
    parser.add_argument(
        "--dtype",
        choices=("f16", "f32"),
        default="f16",
        help="adapter factor storage type",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    args = parser.parse_args()

    try:
        count, scale = convert(
            args.adapter,
            args.output,
            read_alpha(args.adapter, args.alpha),
            args.dtype,
            args.force,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output} ({count} projections, scale {scale:.4g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
