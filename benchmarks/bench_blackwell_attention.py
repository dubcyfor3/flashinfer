"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch
import argparse

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

import cutlass.cute as cute
import math


def bench_fmha_blackwell(
    batch_size,
    qkv_len,
    num_heads,
    head_dim,
    causal,
    dtype,
):
    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    kv_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=dtype, device="cuda"),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        num_heads,
        num_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o = wrapper.run(q, k, v)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q, k, v),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    def flops(ms):
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2 / ms / 1e9
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4 / ms / 1e9

    def io(ms):
        mem_size = q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size() + o.numel() * o.element_size()
        return mem_size / ms / 1e6

    print(
        f"bench_fmha_blackwell (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), flops: {flops(ms):.3f} TFLOPs/s, io: {io(ms):.3f} GB/s"
    )


def bench_fmha_cutedsl(
    batch_size,
    qkv_len,
    num_heads,
    head_dim,
    causal,
    dtype,
    sm_scale=None,
):
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qkv_len
    )

    @cute.jit
    def sigmoid_logits_transform(params, x, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx):
        scale = params.scale
        bias = params.bias
        return cute.arch.rcp_approx(1 + cute.arch.exp2(-(x * scale + bias)))

    @cute.jit
    def dumb_output_transform(params, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * 2.0 * rcp_d

    num_qo_heads = num_heads
    @cute.jit
    def sink_M_D_update(params, kv_tile_idx, qo_head_idx, m, d, scale):
        log_sink = params.sink[qo_head_idx] * math.log2(math.exp(1.0)) if (kv_tile_idx == 0 and qo_head_idx < num_qo_heads) else -math.inf
        m_new = log_sink if log_sink > m else m
        scale = cute.arch.exp2(m - m_new)
        d_new = cute.arch.exp2(log_sink - m_new) + d * scale
        return m_new, d_new
    
    @cute.jit
    def sink_output_transform(params, output, batch_idx, qo_idx, qo_head_idx, m, rcp_d, scale):
        return output * scale * rcp_d

    sink = torch.randn((num_qo_heads,), dtype=dtype, device="cuda")
    
    wrapper = flashinfer.BatchPrefillCuteDSLWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8),
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_transform=sink_output_transform,
        M_D_update=sink_M_D_update,
        use_attention_sink=True,
    )
    o = wrapper.run(q, k, v, sink=sink)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q, k, v, sink=sink),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    def flops(ms):
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2 / ms / 1e9
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4 / ms / 1e9

    def io(ms):
        mem_size = q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size() + o.numel() * o.element_size()
        return mem_size / ms / 1e6

    print(
        f"bench_fmha_cutedsl (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), flops: {flops(ms):.3f} TFLOPs/s, io: {io(ms):.3f} GB/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Blackwell attention implementations")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for benchmarking")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (qkv_len) for benchmarking")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--causal", action="store_true", help="Whether to use causal attention")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", 
                       help="Data type for tensors")
    parser.add_argument("--backend", choices=["cutedsl", "cutlass"], default="cutedsl", 
                       help="Backend to benchmark (cutedsl or cutlass)")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Running benchmark with:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  head_dim: {args.head_dim}")
    print(f"  causal: {args.causal}")
    print(f"  dtype: {args.dtype}")
    print(f"  backend: {args.backend}")
    print()
    
    if args.backend == "cutedsl":
        bench_fmha_cutedsl(
            args.batch_size, 
            args.seq_len, 
            args.num_heads, 
            args.head_dim, 
            args.causal, 
            dtype
        )
    elif args.backend == "cutlass":
        bench_fmha_blackwell(
            args.batch_size, 
            args.seq_len, 
            args.num_heads, 
            args.head_dim, 
            args.causal, 
            dtype
        )

    # bench_fmha_blackwell(128, 512, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(64, 1024, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(32, 2048, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(16, 4096, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(8, 8192, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(4, 16384, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(2, 32768, 32, 128, True, torch.bfloat16)
    # bench_fmha_blackwell(1, 65536, 32, 128, True, torch.bfloat16)

    # Benchmark CuteDSL FMHA
    # print("\n=== CuteDSL FMHA Benchmarks ===")
    # bench_fmha_cutedsl(128, 512, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(64, 1024, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(32, 2048, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(16, 4096, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(8, 8192, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(4, 16384, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(2, 32768, 32, 128, False, torch.bfloat16)
    # bench_fmha_cutedsl(1, 65536, 32, 128, False, torch.bfloat16)

    # bench_fmha_cutedsl(128, 512, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(64, 1024, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(32, 2048, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(16, 4096, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(8, 8192, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(4, 16384, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(2, 32768, 32, 128, True, torch.bfloat16)
    # bench_fmha_cutedsl(1, 65536, 32, 128, True, torch.bfloat16)
