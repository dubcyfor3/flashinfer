# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import enum
import math
import time
from typing import Type, Tuple, Optional, Union, Callable, overload, Literal
from types import SimpleNamespace

import torch
from torch._prims.executor import P
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

"""
A multi-latent attention (MLA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of inference of multi-latent attention using a TMA + Blackwell
SM100 TensorCore warp-specialized persistent kernel. The implementation integrates the (Qc + Qr)*(Kc + Kr)^T
matrix multiplication, softmax normalization, and softmax((Qc + Qr)*(Kc + Kr)^T)*Vc into a single kernel.
The kernel provides support for page table storage and variable-length KV cache sequences. It implements KV splitting 
functionality to minimize latency when processing long KV sequences.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Two sub-kernels (split KV kernel and reduction kernel) that enable split KV processing

To run this example:

.. code-block:: bash

    python examples/blackwell/mla.py                                     \
      --batch_size 4 --latent_dim 512 --rope_dim 64                      \
      --num_heads 128 --seq_len 1024                                     \
      --in_dtype Float8E4M3FN --out_dtype Float16                        \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --use_page_table --is_var_seq --is_var_split_kv                    \
      --is_persistent

The above example runs Multi-Latent Attention (MLA) with the following configuration:
- Batch size: 4
- Sequence length: 1024 
- Latent dimension: 512
- RoPE dimension: 64
- Number of heads: 128
- Data types: Float8E4M3FN (input), Float16 (output), Float32 (accumulation and LSE)

It utilizes page table storage for the KV cache and enables both variable-length KV cache sequences
and variable split KV processing with persistent scheduling.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/mla.py                                 \
      --batch_size 4 --latent_dim 512 --rope_dim 64                      \
      --num_heads 128 --seq_len 1024                                     \
      --in_dtype Float8E4M3FN --out_dtype Float16                        \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --use_page_table --is_var_seq --is_var_split_kv                    \
      --is_persistent --warmup_iterations 3                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Data type requirements:
  - Input/output: Float8E4M3FN or Float16
  - Accumulation and LSE: Float32
* Fixed architecture parameters:
  - Number of attention heads: 128
  - Latent dimension: 512
  - RoPE dimension: 64
* Query sequence length must be 1
* Only supports 2-CTA instructions
* Variable sequence length requires page table storage enabled
"""


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.split_kv)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[1],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            self.cluster_shape_mnk,
            split_kv,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, cluster_shape_mnk, split_kv
    )


class MLAStaticTileScheduler:
    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    1,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            # Used for persistent scheduling
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            blk_coord = self.persistent_blk_layout.get_hier_coord(
                self.current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = (self.blk_coord[0], 0, self.blk_coord[1], self.blk_coord[2])

        return utils.WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 9
        new_params = cutlass.new_from_mlir_values(self.params, values[0:2])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[2]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[3:6])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[6:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


LOG2_E = 1.4426950408889634074

class BlackwellMultiLatentAttentionForward:
    def __init__(
        self,
        latent_dim: int,
        rope_dim: int,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        is_persistent: bool,
        is_cpasync: bool,
        use_page_table: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
    ):
        """Initializes the configuration for a Blackwell Multi-Latent Attention (MLA) kernel.

        :param latent_dim: Latent dimension size
        :type latent_dim: int
        :param rope_dim: RoPE dimension size
        :type rope_dim: int
        :param acc_dtype: Data type for accumulation S and O
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: Data type for output LSE
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_s_tiler: The (H, K) tile shape of the MMA instruction for S
        :type mma_s_tiler: Tuple[int, int]
        :param mma_p_tiler: The (H, D) tile shape of the MMA instruction for P
        :type mma_p_tiler: Tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: int
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param is_cpasync: Whether to use CP async mode
        :type is_cpasync: bool
        :param use_page_table: Whether to use page table
        :type use_page_table: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split KV
        :type is_var_split_kv: bool
        """

        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.is_persistent = is_persistent
        self.is_cpasync = is_cpasync
        self.use_page_table = use_page_table
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.cluster_shape_mnk = (2, 1, 1)
        self.use_2cta_instrs = True
        # When using 2 CTAs with m=128: warps 0-1 handle accumulation for first half [0, n/2),
        # while warps 2-3 handle accumulation for second half [n/2, n)
        self.warps_in_n = 2
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        self.num_load_warps = 2 if self.is_cpasync else 1
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            self.rope_dim,
        )
        self.mma_pv_tiler = (self.mma_pv_tiler_mn[0], self.mma_pv_tiler_mn[1], 32)
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk_rope = self.rope_dim // self.mma_qk_tiler[2]
        self.iterations_qk = self.iterations_qk_latent + self.iterations_qk_rope
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        self.iterations_pv_n = self.latent_dim // self.mma_pv_tiler[1]

        # Set specialized warp ids
        self.compute_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        if self.is_cpasync:
            self.load_cp_async_warp_ids = (5, 6)
            self.load_pt_warp_id = 7
            self.threads_per_cta = self.threads_per_warp * len(
                (
                    self.mma_warp_id,
                    *self.load_cp_async_warp_ids,
                    self.load_pt_warp_id,
                    *self.compute_warp_ids,
                )
            )
        else:
            self.load_tma_warp_id = 5
            self.threads_per_cta = self.threads_per_warp * len(
                (self.mma_warp_id, self.load_tma_warp_id, *self.compute_warp_ids)
            )

        # Named barriers
        self.tmem_ptr_sync_bar_id = 0
        self.exchange_sync_bar_id = 1
        self.epilog_sync_bar_id = 2
        self.cta_sync_bar_id = 3


    def _setup_attributes(self):
        """Set up configurations and parameters for the MLA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the multi-latent attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.load_qkv_stage = 24 // (self.q_dtype.width // 8)
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.mma_o_stage = 1
        self.load_pt_stage = self.load_qkv_stage if self.is_cpasync else 1

        self.tmem_o_offset = self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Multi-Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, batch_size]
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [num_head, rope_dim, batch_size]
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [seq_len, latent_dim, batch_size]
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [seq_len, rope_dim, batch_size]
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [page_count, batch_size]
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype:
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            split_kv,
            self.acc_dtype,
            workspace,
        )

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self.q_major_mode = tcgen05.OperandMajorMode.K
        self.k_major_mode = tcgen05.OperandMajorMode.K
        self.v_major_mode = tcgen05.OperandMajorMode.MN

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            self.iterations_qk,
        )
        kc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            self.load_qkv_stage,
        )
        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )
        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            self.load_qkv_stage,
        )

        # TMA load for Q latent and rope
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_rope,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent and k rope
        kc_smem_layout = cute.select(kc_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_c_latent, tma_tensor_c_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            c_latent,
            kc_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        tma_atom_c_rope, tma_tensor_c_rope = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            c_rope,
            kc_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent transpose
        vc_smem_layout = cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                c_latent_transpose,
                vc_smem_layout,
                self.mma_pv_tiler,
                pv_tiled_mma,
                cta_layout_vmnk.shape,
            )
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )
        kc_copy_size = cute.size_in_bytes(self.k_dtype, kc_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )

        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kc_bytes = kc_copy_size

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
        )

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            load_qkv_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_qkv_stage * 2
            ]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]
            load_pt_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_pt_stage * 2
            ]

            # Smem tensors
            smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

            smem_page_table: cute.struct.MemRange[
                cutlass.Int32, self.load_pt_stage * self.mma_qk_tiler[1]
            ]
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                1024,
            ]
            smem_kc: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kc_smem_layout_staged)],
                1024,
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: cutlass.Int64

            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32


        softmax_scale_log2 = softmax_scale * LOG2_E
        # Launch the kernel synchronously
        self.split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_q_rope,
            tma_tensor_q_rope,
            tma_atom_c_latent,
            tma_tensor_c_latent,
            tma_atom_c_rope,
            tma_tensor_c_rope,
            tma_atom_c_latent_transpose,
            tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            q_smem_layout_staged,
            kc_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            cta_layout_vmnk,
            tile_sched_params,
            SplitKVKernelSharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SplitKVKernelSharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        if acc_o is not None:
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], 1, q_latent.shape[2]),
                block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                smem=split_kv * self.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: cute.CopyAtom,
        mQL: cute.Tensor,
        tma_atom_q_rope: cute.CopyAtom,
        mQR: cute.Tensor,
        tma_atom_c_latent: cute.CopyAtom,
        mCL: cute.Tensor,
        tma_atom_c_rope: cute.CopyAtom,
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: cute.CopyAtom,
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        """The device split_kv kernel implementation of the Multi-Latent Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the MLA computation:
        1. Load warp: Loads Q/C latent/rope data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the MLA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param tma_atom_q_rope: TMA copy atom for query rope tensor
        :type tma_atom_q_rope: cute.CopyAtom
        :param mKR: Compressed rope tensor
        :type mKR: cute.Tensor
        :param tma_atom_c_latent: TMA copy atom for c latent tensor
        :type tma_atom_c_latent: cute.CopyAtom
        :param mCL: Compressed latent tensor
        :type mCL: cute.Tensor
        :param tma_atom_c_rope: TMA copy atom for c rope tensor
        :type tma_atom_c_rope: cute.CopyAtom
        :param mCLT: Compressed latent transpose tensor
        :type mCLT: cute.Tensor
        :param mPT: Page table tensor
        :type mPT: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The variable sequence length tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param kc_smem_layout_staged: Shared memory layout for key tensor
        :type kc_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: MLAStaticTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Coords inside cluster
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        # Prefetch tma descriptor
        if cutlass.dynamic_expr(warp_idx == 0):
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Tensor memory dealloc barrier init
        if cutlass.dynamic_expr(warp_idx == self.load_tma_warp_id):
            num_tmem_dealloc_threads = self.threads_per_warp * self.num_compute_warps
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        load_qkv_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_qkv_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        if self.is_cpasync:
            load_pt_pipeline = self.make_and_init_load_pt_pipeline(
                storage.load_pt_mbar_ptr.data_ptr()
            )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc.get_tensor(
            kc_smem_layout_staged.outer, swizzle=kc_smem_layout_staged.inner
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        # reuse smem
        sVC_ptr = cute.recast_ptr(sKC.iterator, vc_smem_layout_staged.inner)
        sVC = cute.make_tensor(sVC_ptr, vc_smem_layout_staged.outer)
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (compute_threads,)
        smem_exchange = storage.smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        if self.is_cpasync:
            # TODO: add cp async load variant.
            #  Load page table when isasync is true
            # if warp_idx == self.load_pt_warp_id:
            #     self.load_page_table()
            # if (
            #     warp_idx == self.load_cpasync_warp_id[0]
            #     and warp_idx == self.load_cpasync_warp_id[1]
            # ):
            #     load_cpasync()
            pass
        else:
            if cutlass.dynamic_expr(warp_idx == self.load_tma_warp_id):
                load_qkv_producer_state = utils.make_pipeline_state(
                    utils.PipelineUserType.Producer, self.load_qkv_stage
                )
                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while cutlass.dynamic_expr(work_tile.is_valid_tile):
                    blk_coord = work_tile.tile_idx
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv,
                        cache_seqs,
                        block_split_kvs,
                        blk_coord,
                    )
                    if cutlass.dynamic_expr(k_tile_count > 0):
                        # Construct fixed common/tma_qk/tma_pv params for load_tma
                        tma_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            local_split_kv=local_split_kv,
                            load_qkv_pipeline=load_qkv_pipeline,
                            mPT=mPT,
                        )
                        tma_qk_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            tma_atom_q_latent=tma_atom_q_latent,
                            tma_atom_q_rope=tma_atom_q_rope,
                            tma_atom_c_latent=tma_atom_c_latent,
                            tma_atom_c_rope=tma_atom_c_rope,
                            mQL=mQL,
                            mQR=mQR,
                            mCL=mCL,
                            mKR=mKR,
                            mCLT=mCLT,
                            sQ=sQ,
                            sKC=sKC,
                        )
                        tma_pv_params = SimpleNamespace(
                            tiled_mma_pv=tiled_mma_pv,
                            tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                            mCL=mCL,
                            mKR=mKR,
                            mCLT=mCLT,
                            sVC=sVC,
                        )
                        # Load tma
                        load_qkv_producer_state = self.load_tma(
                            tma_common_params,
                            tma_qk_params,
                            tma_pv_params,
                            k_index,
                            k_tile_count,
                            load_qkv_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                    # sync with compute warp before next round
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=self.threads_per_warp
                        * (self.num_compute_warps + self.num_load_warps),
                    )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.dynamic_expr(warp_idx == self.mma_warp_id):
            # Alloc tensor memory buffer
            cute.arch.alloc_tmem(
                cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
                tmem_holding_buf,
                is_two_cta=self.use_2cta_instrs,
            )

            # warp synchronization
            # TODO: is this warp sync necessary?
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=self.threads_per_warp,
            )

            # Retrieving tensor memory ptr and make accumulator tensor
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

            load_qkv_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.load_qkv_stage
            )
            mma_s_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.mma_s_stage
            )
            p_mma_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.mma_o_stage
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while cutlass.dynamic_expr(work_tile.is_valid_tile):
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if cutlass.dynamic_expr(k_tile_count > 0):
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_qkv_pipeline=load_qkv_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sKC=sKC,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_qkv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma(
                        mma_common_params,
                        mma_qk_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_qkv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_o_pipeline.producer_tail(mma_o_producer_state)

            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta_instrs)
            # Dealloc the tensor memory buffer
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)

            cute.arch.dealloc_tmem(
                tmem_ptr,
                cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
                is_two_cta=self.use_2cta_instrs,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.dynamic_expr(
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            mma_s_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Producer, self.p_mma_stage
            )
            mma_o_consumer_state = utils.make_pipeline_state(
                utils.PipelineUserType.Consumer, self.mma_o_stage
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while cutlass.dynamic_expr(work_tile.is_valid_tile):
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if cutlass.dynamic_expr(k_tile_count > 0):
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        L=mCL.shape[1],
                        tidx=tidx,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    compute_rescale_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        mma_o_pipeline=mma_o_pipeline,
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            compute_rescale_params,
                            compute_epilogue_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            mma_o_consumer_state=mma_o_consumer_state,
                        )
                    )
                else:
                    cute.arch.barrier_arrive(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=(self.num_compute_warps + self.num_load_warps)
                        * self.threads_per_warp,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Arrive for the tensor memory deallocation barrier
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
    ):
        """The reduction kernel for Multi-Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        """
        # avoid register indexing on array.
        MAX_SPLITS = 256
        bidx, _, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, 0, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.dynamic_expr(warp_idx == 0):
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.threads_per_warp)

            local_lse = cute.make_fragment(
                cute.make_layout(lse_per_thread), self.lse_dtype
            )
            lse_max = -self.lse_dtype.inf
            # find the max lse
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = self._warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in range(lse_per_thread):
                sum_lse += cute.arch.exp2(local_lse[i] - lse_max)
            sum_lse = self._warp_reduction_sum(sum_lse)
            # calculate the global_lse
            global_lse = (
                lse_max + cute.arch.log2(sum_lse)
                if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse
                else self.lse_dtype.inf
            )
            if cutlass.dynamic_expr(tidx == 0):
                mLSE[blk_coord[0], blk_coord[2]] = global_lse
            # store the scale to shared memory
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                if cutlass.dynamic_expr(cute.elem_less(split_kv_idx, local_split_kv)):
                    smem_lse_scale[split_kv_idx] = cute.arch.exp2(
                        local_lse[i] - global_lse
                    )

        cute.arch.barrier()

        elements_per_thread = cute.ceil_div(
            self.latent_dim, self.threads_per_warp * self.num_compute_warps
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[2]]
        rAccO = cute.make_fragment(
            cute.make_layout(elements_per_thread), self.acc_dtype
        )
        rAccO.fill(0.0)
        for i in range_dynamic(local_split_kv):
            for j in range(elements_per_thread):
                element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
                rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
        for j in range(elements_per_thread):
            element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
            mO[blk_coord[0], element_idx, blk_coord[2]] = rAccO[j].to(self.o_dtype)
        return

    @staticmethod
    def get_split_kv(
        B: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
    ) -> int:
        """Get the proper split_kv value for the MLA kernel based on parameters.

        :param B: Batch size
        :type B: int
        :param K: Sequence length
        :type K: int
        :param mma_qk_tiler_mn: MLA tiling parameters
        :type mma_qk_tiler_mn: tuple
        :param max_active_blocks: Maximum number of active blocks
        :type max_active_blocks: int
        :return: Split_kv value
        :rtype: int
        """
        max_splits = ceil_div(K, mma_qk_tiler_mn[1])
        blocks_per_batch = max(1, max_active_blocks // B)
        split_heur = min(max_splits, blocks_per_batch)
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_waves = ceil_div(max_splits, split_heur)
        split_wave_aware = ceil_div(max_splits, k_waves)
        return split_wave_aware

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the MLA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[2]]
        if self.is_var_split_kv:
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def load_tma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_qkv_producer_state: utils.PipelineState,
    ) -> utils.PipelineState:
        """Load wrap to load Q/C latent/rope tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_qkv_producer_state: The load qkv producer state
        :type load_qkv_producer_state: utils.PipelineState

        :return: The load qkv producer state
        :rtype: utils.PipelineState
        """
        # page table
        mPT = None
        if self.use_page_table:
            mPT = common_params.mPT[None, common_params.blk_coord[2]]

        # Flatten divide and partition global tensors for QK TMA load
        # (bM, bK, rM, rK, rL)
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)
        gQR = cute.flat_divide(qk_params.mQR, mma_qk_tiler_mk)

        mma_qk_tiler_nk = cute.select(self.mma_qk_tiler, mode=[1, 2])
        gCL = cute.flat_divide(qk_params.mCL, mma_qk_tiler_nk)
        gKR = cute.flat_divide(qk_params.mKR, mma_qk_tiler_nk)

        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)

        tSgCL = thr_mma_qk.partition_B(gCL)
        tSgKR = thr_mma_qk.partition_B(gKR)

        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        _, tQRgQR_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )

        tKCsKC, tCLgCL = cpasync.tma_partition(
            qk_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sKC, 0, 3),
            cute.group_modes(tSgCL, 0, 3),
        )

        _, tKRgKR = cpasync.tma_partition(
            qk_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sKC, 0, 3),
            cute.group_modes(tSgKR, 0, 3),
        )

        tQLgQL = tQLgQL_mkl[None, None, None, common_params.blk_coord[2]]
        tQRgQR = tQRgQR_mkl[None, None, None, common_params.blk_coord[2]]

        # Flatten divide and partition global tensors for V TMA load
        mma_pv_tiler_nk = cute.select(self.mma_pv_tiler, mode=[1, 2])
        gCLT = cute.flat_divide(v_params.mCLT, mma_pv_tiler_nk)

        thr_mma_pv = v_params.tiled_mma_pv.get_slice(
            common_params.blk_coord[0] % cute.size(v_params.tiled_mma_pv.thr_id)
        )
        tOgCLT = thr_mma_pv.partition_B(gCLT)

        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            cute.group_modes(v_params.sVC, 0, 3),
            cute.group_modes(tOgCLT, 0, 3),
        )

        # set extra params
        common_params.mPT = mPT
        qk_params.tQLgQL = tQLgQL
        qk_params.tQRgQR = tQRgQR
        qk_params.tCLgCL = tCLgCL
        qk_params.tKRgKR = tKRgKR
        qk_params.tQsQ = tQsQ
        qk_params.tKCsKC = tKCsKC
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

        # {$nv-internal-release begin}
        # TODO: simplify the code after software pipeline is turning on by default.
        # TODO: add prefetch for next k tile
        # {$nv-internal-release end}
        load_qkv_producer_state = self.load_tma_qk_one_k_tile(
            common_params,
            qk_params,
            k_index,
            load_qkv_producer_state,
            load_q=True,
        )
        k_index += 1
        k_tile_count -= 1
        while cutlass.dynamic_expr(k_tile_count > 0):
            # {$nv-internal-release begin}
            # TODO: figure out how to support SingleNamespace/struct in ast
            # {$nv-internal-release end}
            load_qkv_producer_state = self.load_tma_qk_one_k_tile(
                common_params,
                qk_params,
                k_index,
                load_qkv_producer_state,
                load_q=False,
            )
            load_qkv_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index,
                load_qkv_producer_state,
            )
            k_index += 1
            k_tile_count -= 1

        # load last v tile
        load_qkv_producer_state = self.load_tma_v_one_k_tile(
            common_params,
            v_params,
            k_index,
            load_qkv_producer_state,
        )
        return load_qkv_producer_state

    @cute.jit
    def load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_qkv_producer_state: utils.PipelineState,
        load_q: bool,
    ) -> utils.PipelineState:
        """Load one k-tile of Q/C latent/rope tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_qkv_producer_state: The load qkv producer state
        :type load_qkv_producer_state: utils.PipelineState
        :param load_q: Whether to load q
        :type load_q: bool

        :return: The load qkv producer state
        :rtype: utils.PipelineState
        """
        for i in range(self.iterations_qk_latent):
            # get the mbar ptr from pipeline.
            tma_bar_ptr = common_params.load_qkv_pipeline.producer_get_barrier(
                load_qkv_producer_state
            )
            # expect the extra bytes for q.
            common_params.load_qkv_pipeline.producer_acquire(
                load_qkv_producer_state,
                extra_expect_tx=(self.tma_copy_q_bytes if load_q else 0),
            )
            # load q once at first iteration
            if load_q:
                # load q latent
                cute.copy(
                    qk_params.tma_atom_q_latent,
                    qk_params.tQLgQL[None, 0, i],
                    qk_params.tQsQ[None, i],
                    tma_bar_ptr=tma_bar_ptr,
                )
            # load k latent
            if self.use_page_table:
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, 0, i, common_params.mPT[k_index]],
                    qk_params.tKCsKC[None, load_qkv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            else:
                cute.copy(
                    qk_params.tma_atom_c_latent,
                    qk_params.tCLgCL[None, k_index, i, common_params.blk_coord[2]],
                    qk_params.tKCsKC[None, load_qkv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_qkv_producer_state.advance()

        for i in range(self.iterations_qk_rope):
            # get the mbar ptr from pipeline.
            tma_bar_ptr = common_params.load_qkv_pipeline.producer_get_barrier(
                load_qkv_producer_state
            )
            # expect the extra bytes for q.
            common_params.load_qkv_pipeline.producer_acquire(
                load_qkv_producer_state,
                extra_expect_tx=(self.tma_copy_q_bytes if load_q else 0),
            )
            if load_q:
                # load q rope
                cute.copy(
                    qk_params.tma_atom_q_rope,
                    qk_params.tQRgQR[None, 0, i],
                    qk_params.tQsQ[None, i + self.iterations_qk_latent],
                    tma_bar_ptr=tma_bar_ptr,
                )
            # load k rope
            if self.use_page_table:
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, 0, i, common_params.mPT[k_index]],
                    qk_params.tKCsKC[None, load_qkv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            else:
                cute.copy(
                    qk_params.tma_atom_c_rope,
                    qk_params.tKRgKR[None, k_index, i, common_params.blk_coord[2]],
                    qk_params.tKCsKC[None, load_qkv_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_qkv_producer_state.advance()
        return load_qkv_producer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_qkv_producer_state: utils.PipelineState,
    ) -> utils.PipelineState:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_qkv_producer_state: The load qkv producer state
        :type load_qkv_producer_state: utils.PipelineState

        :return: The load qkv producer state
        :rtype: utils.PipelineState
        """
        for i in range(self.iterations_pv_k):
            for j in range(self.iterations_pv_n):
                # get the mbar ptr from pipeline.
                tma_bar_ptr = common_params.load_qkv_pipeline.producer_get_barrier(
                    load_qkv_producer_state
                )
                common_params.load_qkv_pipeline.producer_acquire(
                    load_qkv_producer_state
                )
                if self.use_page_table:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[None, j, i, common_params.mPT[k_index - 1]],
                        v_params.tVCsVC[None, load_qkv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                else:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[
                            None,
                            j,
                            (k_index - 1) * self.iterations_pv_k + i,
                            common_params.blk_coord[2],
                        ],
                        v_params.tVCsVC[None, load_qkv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                load_qkv_producer_state.advance()
        return load_qkv_producer_state

    @cute.jit
    def mma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_qkv_consumer_state: utils.PipelineState,
        mma_s_producer_state: utils.PipelineState,
        p_mma_consumer_state: utils.PipelineState,
        mma_o_producer_state: utils.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        utils.PipelineState,
        utils.PipelineState,
        utils.PipelineState,
    ]:
        """MMA warp to compute the result of Q*K^T and P*V. Updates the tiled mma and pipeline states.

        :param common_params: The common parameters for mma qk and pv
        :type common_params: SimpleNamespace
        :param qk_params: The mma qk parameters
        :type qk_params: SimpleNamespace
        :param pv_params: The mma pv parameters
        :type pv_params: SimpleNamespace
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_qkv_consumer_state: The load qkv consumer state
        :type load_qkv_consumer_state: utils.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: utils.PipelineState
        :param p_mma_consumer_state: The p mma consumer state
        :type p_mma_consumer_state: utils.PipelineState
        :param mma_o_producer_state: The mma o producer state
        :type mma_o_producer_state: utils.PipelineState

        :return: The tiled mma qk, the tiled mma pv, the load qkv consumer state, the mma s producer state, the p mma consumer state, and the mma o producer state
        :rtype: tuple[cute.TiledMma, cute.TiledMma, utils.PipelineState, utils.PipelineState, utils.PipelineState, utils.PipelineState]
        """

        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        # mma O has 1 stage.
        assert (
            self.mma_o_stage == 1
        ), "mma O has 1 stage, otherwise the tmem usage exceeds the limit."
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO_staged = cute.make_tensor(
            tStS_staged.iterator + self.tmem_o_offset, tOtO_layout
        )

        # set more parameters
        qk_params.tSrQ = tSrQ
        qk_params.tSrKC = tSrKC
        qk_params.tStS_staged = tStS_staged
        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        # mma O accumulates on K, so the accumlate flag is set to False once before all K blocks.
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        if cutlass.dynamic_expr(common_params.is_leader_cta):
            tiled_mma_qk, load_qkv_consumer_state, mma_s_producer_state = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                load_qkv_consumer_state,
                mma_s_producer_state,
            )
            k_tile_count -= 1

            while cutlass.dynamic_expr(k_tile_count > 0):
                tiled_mma_qk, load_qkv_consumer_state, mma_s_producer_state = (
                    self.mma_qk(
                        common_params,
                        qk_params,
                        tiled_mma_qk,
                        load_qkv_consumer_state,
                        mma_s_producer_state,
                    )
                )
                (
                    tiled_mma_pv,
                    load_qkv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    tiled_mma_pv,
                    load_qkv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                k_tile_count -= 1

            (
                tiled_mma_pv,
                load_qkv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            ) = self.mma_pv(
                common_params,
                pv_params,
                tiled_mma_pv,
                load_qkv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            )

        return (
            tiled_mma_qk,
            tiled_mma_pv,
            load_qkv_consumer_state,
            mma_s_producer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_qkv_consumer_state: utils.PipelineState,
        mma_s_producer_state: utils.PipelineState,
    ) -> tuple[cute.TiledMma, utils.PipelineState, utils.PipelineState]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_qkv_consumer_state: The load qkv consumer state
        :type load_qkv_consumer_state: utils.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: utils.PipelineState

        :return: The tiled mma qk, the load qkv consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, utils.PipelineState, utils.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        for q_stage in range(self.iterations_qk):
            common_params.load_qkv_pipeline.consumer_wait(load_qkv_consumer_state)
            kc_stage = load_qkv_consumer_state.index
            for k_block in range(qk_params.tSrQ.shape[2]):
                cute.gemm(
                    tiled_mma_qk,
                    tStS,
                    qk_params.tSrQ[None, None, k_block, q_stage],
                    qk_params.tSrKC[None, None, k_block, kc_stage],
                    tStS,
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            common_params.load_qkv_pipeline.consumer_release(load_qkv_consumer_state)
            load_qkv_consumer_state.advance()
        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return tiled_mma_qk, load_qkv_consumer_state, mma_s_producer_state

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        load_qkv_consumer_state: utils.PipelineState,
        p_mma_consumer_state: utils.PipelineState,
        mma_o_producer_state: utils.PipelineState,
    ) -> tuple[
        cute.TiledMma, utils.PipelineState, utils.PipelineState, utils.PipelineState
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_qkv_consumer_state: The load qkv consumer state
        :type load_qkv_consumer_state: utils.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: utils.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: utils.PipelineState

        :return: The tiled mma pv, the load qkv consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, utils.PipelineState, utils.PipelineState, utils.PipelineState]
        """

        pv_params.mma_o_pipeline.producer_acquire(mma_o_producer_state)
        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        for p_stage in range(self.iterations_pv_k):
            accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
            for acc_stage in range(self.iterations_pv_n):
                common_params.load_qkv_pipeline.consumer_wait(load_qkv_consumer_state)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                vc_stage = load_qkv_consumer_state.index
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                for k_block in range(pv_params.tOrP.shape[2]):
                    cute.gemm(
                        tiled_mma_pv,
                        tOtO,
                        pv_params.tOrP[
                            None,
                            None,
                            k_block,
                            (p_stage, p_mma_consumer_state.index),
                        ],
                        pv_params.tOrVC[None, None, k_block, vc_stage],
                        tOtO,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                common_params.load_qkv_pipeline.consumer_release(
                    load_qkv_consumer_state
                )
                load_qkv_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()
        pv_params.mma_o_pipeline.producer_commit(mma_o_producer_state)
        mma_o_producer_state.advance()

        return (
            tiled_mma_pv,
            load_qkv_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        rescale_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: utils.PipelineState,
        p_mma_producer_state: utils.PipelineState,
        mma_o_consumer_state: utils.PipelineState,
    ) -> tuple[utils.PipelineState, utils.PipelineState, utils.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param rescale_params: The rescale parameters
        :type rescale_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: utils.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: utils.PipelineState
        :param mma_o_consumer_state: The MMA o consumer state
        :type mma_o_consumer_state: utils.PipelineState

        :return: The MMA s consumer state, the P MMA producer state, and the MMA o consumer state
        :rtype: tuple[utils.PipelineState, utils.PipelineState, utils.PipelineState]
        """

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)

        (
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max,
            row_sum,
            correction_factor,
        ) = self.softmax(
            common_params,
            softmax_params,
            k_index,
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max,
            row_sum,
            correction_factor,
            k_index == k_tile_total - 1,
        )
        k_index = k_index + 1
        k_tile_count = k_tile_count - 1

        while cutlass.dynamic_expr(k_tile_count > 0):
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
                k_index == k_tile_total - 1,
            )
            mma_o_consumer_state = self.rescale(
                common_params, rescale_params, mma_o_consumer_state, correction_factor
            )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        mma_o_consumer_state = self.epilogue(
            common_params, epilogue_params, mma_o_consumer_state, row_max, row_sum
        )
        return mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: utils.PipelineState,
        p_mma_producer_state: utils.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_last_tile: cutlass.Boolean,
    ) -> tuple[
        utils.PipelineState,
        utils.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Softmax for one k-tile. Updates the related pipeline states and returns the computed results.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: utils.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: utils.PipelineState
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param is_last_tile: Whether the last tile
        :type is_last_tile: cutlass.Boolean

        :return: The MMA s consumer state, the P MMA producer state, the row max, the row sum, and the correction factor
        :rtype: tuple[utils.PipelineState, utils.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32]
        """

        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        softmax_params.mma_s_pipeline.consumer_wait(mma_s_consumer_state)

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)
        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)

        row_max_new = row_max
        for i in range(cute.size(tTR_rAcc)):
            if cutlass.dynamic_expr(is_last_tile):
                tTR_rAcc[i] = (
                    -self.acc_dtype.inf
                    if not cute.elem_less(
                        tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                        common_params.K,
                    )
                    else tTR_rAcc[i]
                )
            # update row_max
            row_max_new = cute.arch.fmax(row_max_new, tTR_rAcc[i])

        # if warps in N is 2, reduce row_max across warps (0, 1) and (2, 3)
        if self.warps_in_n == 2:
            common_params.smem_exchange[tidx] = row_max_new
            cute.arch.barrier(
                barrier_id=self.exchange_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )

        # find correction factor
        correction_factor = self._exp2f(
            (row_max - row_max_new) * softmax_params.softmax_scale_log2
        )
        # update row_max
        row_max = row_max_new
        # softmax
        tTR_rAcc_vec = tTR_rAcc.load()
        tTR_rAcc_vec = cute.TensorSSA(
            self._exp2f(
                tTR_rAcc_vec * softmax_params.softmax_scale_log2
                - row_max_new * softmax_params.softmax_scale_log2
            ),
            tuple(tTR_rAcc_vec.shape),
            cutlass.Float32,
        )

        # quantize
        tTR_rS.store(tTR_rAcc_vec.to(self.q_dtype))

        # create sP
        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        # change to PISL
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = 2 if self.q_dtype.width == 16 else 1
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP = cute.make_tensor(
            sP_wo_swizzle_iter, cute.make_composed_layout(sP_swizzle, 0, sP.layout)
        )
        l = cute.make_ordered_layout(
            ((64, 2), (16, self.mma_qk_tiler[1] // 32)), order=((0, 3), (1, 2))
        )
        sP_copy_view = cute.composition(sP, l)
        cute.autovec_copy(tTR_rS, sP_copy_view[tidx, None])

        # row_sum, using `add_packed_f32x2` to reduce the number of instructions
        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in range(0, cute.size(tTR_rAcc_vec.shape), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc_vec[i], tTR_rAcc_vec[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        # fence between tmem load and mma s
        cute.arch.fence_view_async_tmem_load()
        # fence between smem store and mma o
        cute.arch.fence_view_async_shared()

        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        mma_s_consumer_state.advance()
        p_mma_producer_state.advance()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def _tmem_load_partition(
        self, common_params: SimpleNamespace, tiled_mma_pv: cute.TiledMma, iter_n: int
    ) -> tuple[
        cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
    ]:
        """Tensor memory load partition for rescale and epilogue."""

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        # {$nv-internal-release begin}
        # TODO: supports size() on tiled copy.
        # {$nv-internal-release end}
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if common_params.mAccO is not None:
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None],
                cta_pv_tiler_mn,
                (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO = tmem_load_thr_copy.partition_D(gO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_rAcc

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        rescale_params: SimpleNamespace,
        mma_o_consumer_state: utils.PipelineState,
        correction_factor: cutlass.Float32,
    ) -> utils.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state."""

        rescale_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)

        for iter_n in range(self.iterations_pv_n):
            # tmem load tiled copy and partition results.
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, rescale_params.tiled_mma_pv, iter_n
                )
            )
            # tmem store tiled copy
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
            tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

            # load o
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            # rescale, using `mul_packed_f32x2` to reduce the number of instructions
            for i in range(0, cute.size(tTR_rAcc), 2):
                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                    (correction_factor, correction_factor),
                )

            # {$nv-internal-release begin}
            # TODO: figure out if we could reduce the sttm for last k tile.
            # {$nv-internal-release end}
            # store o to tensor memory for next k tile
            cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

        cute.arch.fence_view_async_tmem_store()
        rescale_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: utils.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
    ) -> utils.PipelineState:
        """Epilogue for one k-tile. Updates the related pipeline state."""

        # mma_o pipeline consumer wait
        epilogue_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)

        # exchange row_sum between warps (0, 1) and (2, 3)
        if self.warps_in_n == 2:
            common_params.smem_exchange[common_params.tidx] = row_sum
            cute.arch.barrier(
                barrier_id=self.exchange_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (common_params.tidx + 64)
                    % (self.num_compute_warps * self.threads_per_warp)
                ]
            )

        # sync with load warps before next round
        cute.arch.barrier_arrive(
            barrier_id=self.epilog_sync_bar_id,
            number_of_threads=(self.num_compute_warps + self.num_load_warps)
            * self.threads_per_warp,
        )

        for iter_n in range(self.iterations_pv_n):
            # tmem load tiled copy and partition results.
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, epilogue_params.tiled_mma_pv, iter_n
                )
            )

            # load o
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

            # apply output scale and normalize by row_sum
            tTR_rAcc.store(
                tTR_rAcc.load()
                * epilogue_params.output_scale
                * cute.arch.rcp_approx(row_sum)
            )

            # store o to global memory
            tR2G_rO_src = None
            tR2G_rO_dst = tTR_gO
            if common_params.mAccO is None:
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
                # using final output dtype for o
                tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
            else:
                # using accumulate dtype for o
                tR2G_rO_src = tTR_rAcc
            cute.autovec_copy(tR2G_rO_src, tR2G_rO_dst)

            # store the lse to global memory
            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            gLSE = None
            if epilogue_params.mAccLSE is None:
                gLSE = cute.local_tile(
                    epilogue_params.mLSE,
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
            else:
                gLSE = cute.local_tile(
                    epilogue_params.mAccLSE[None, common_params.blk_coord[3], None],
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, None, 1),
                )
            lse = cute.arch.log2(row_sum) + epilogue_params.softmax_scale_log2 * row_max
            if self.warps_in_n == 2:
                if cutlass.dynamic_expr(common_params.tidx < 64):
                    gLSE[common_params.tidx] = lse

        cute.arch.fence_view_async_tmem_load()
        epilogue_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk
    ) -> utils.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline."""

        load_qkv_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread, len([self.load_tma_warp_id])
        )
        load_qkv_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, len([self.mma_warp_id])
        )
        return utils.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=self.load_qkv_stage,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            # tx_count of tma_q_bytes is applied by `producer_expect_transaction` later
            tx_count=self.tma_copy_kc_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> utils.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline."""

        mma_s_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_s_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            consumer_thread_size,
            consumer_thread_size,
        )
        return utils.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def make_and_init_p_mma_pipeline(
        self, p_mma_mbar_ptr, cta_layout_vmnk
    ) -> utils.PipelineAsyncUmma:
        """Create and initialize the p mma pipeline."""

        producer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        p_mma_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            producer_thread_size,
            producer_thread_size,
        )
        p_mma_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread, len([self.mma_warp_id])
        )
        return utils.PipelineAsyncUmma.create(
            barrier_storage=p_mma_mbar_ptr,
            num_stages=self.p_mma_stage,
            producer_group=p_mma_producer_group,
            consumer_group=p_mma_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> utils.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline."""

        mma_o_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_o_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            consumer_thread_size,
            consumer_thread_size,
        )
        return utils.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        )

    def make_and_init_load_pt_pipeline(self, load_pt_mbar_ptr):
        """Create and initialize the load page table pipeline."""
        load_pt_producer_group = utils.CooperativeGroup(
            utils.Agent.Thread, self.threads_per_warp * len([self.load_pt_warp_id])
        )
        load_pt_consumer_group = utils.CooperativeGroup(
            utils.Agent.Thread,
            self.threads_per_warp * len([self.load_cp_async_warp_id]),
        )
        return utils.PipelineAsync.create(
            barrier_storage=load_pt_mbar_ptr,
            num_stages=self.load_pt_stage,
            producer_group=load_pt_producer_group,
            consumer_group=load_pt_consumer_group,
        )

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[MLAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_mla_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = MLAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    def _exp2f(
        self, x: Union[cute.TensorSSA, cutlass.Float32]
    ) -> Union[cute.TensorSSA, cutlass.Float32]:
        """exp2f calculation for both vector and scalar.

        :param x: input value
        :type x: cute.TensorSSA or cutlass.Float32
        :return: exp2 value
        :rtype: cute.TensorSSA or cutlass.Float32
        """
        if isinstance(x, cute.TensorSSA):
            res = cute.make_fragment(x.shape, cutlass.Float32)
            res.store(x)

            for i in range(cute.size(x.shape)):
                res[i] = self._exp2f(res[i])

            return res.load()
        return cute.arch.exp2(x)

    def _warp_reduction(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """warp reduction, accepts custom binary operator.

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        offset = 16
        while offset > 0:
            val = op(
                val,
                cute.arch.shuffle_sync_bfly(
                    val, offset=offset, mask=-1, mask_and_clamp=31
                ),
            )
            offset = offset // 2
        return val

    def _warp_reduction_max(self, val: cutlass.Float32) -> cutlass.Float32:
        """warp reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._warp_reduction(val, lambda x, y: cute.arch.fmax(x, y))

    def _warp_reduction_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        """warp reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._warp_reduction(val, lambda x, y: x + y)

    @staticmethod
    def get_workspace_size(
        H: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Get the extra workspace(device memory) size for the MLA kernel when split_kv is not 1.

        :param H: The height of the output tensor C
        :type H: int
        :param D: The depth of the output tensor C
        :type D: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]

        :return: The workspace size for the MLA kernel
        :rtype: int
        """
        if split_kv == 1:
            return 0
        return B * H * split_kv * (D + 1) * acc_dtype.width // 8

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if workspace is not None:
            align = 128 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(H * split_kv * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, B), stride=(split_kv, 1, H * split_kv)
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator + cute.cosize(acc_o_layout) * acc_dtype.width // 8,
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_cpasync: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        use_page_table: bool,
        page_size: int,
    ) -> bool:
        """Check if the MLA kernel can be implemented.

        :param H: The height of the output tensor C
        :type H: int
        :param K: The width of the output tensor C
        :type K: int
        :param L: The length of the output tensor C
        :type L: int
        :param R: The row of the output tensor C
        :type R: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_cpasync: Whether to use cpasync
        :type is_cpasync: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param use_page_table: Whether to use page table
        :type use_page_table: bool
        :param page_size: The page size of the page table
        :type page_size: int

        :return: Whether the MLA kernel can be implemented
        :rtype: bool
        """
        if in_dtype not in [cutlass.Float8E4M3FN, cutlass.Float16]:
            print(f"[can_implement] Unsupported in_dtype: {in_dtype}. Must be Float8E4M3FN or Float16.")
            return False
        if out_dtype != cutlass.Float16:
            print(f"[can_implement] Unsupported out_dtype: {out_dtype}. Must be Float16.")
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            print(f"[can_implement] acc_dtype or lse_dtype not Float32. acc_dtype: {acc_dtype}, lse_dtype: {lse_dtype}")
            return False
        if is_cpasync:
            if not use_page_table:
                print(f"[can_implement] is_cpasync is True but use_page_table is False.")
                return False
            if page_size & (page_size - 1) != 0:
                print(f"[can_implement] page_size {page_size} is not a power of 2.")
                return False
            if page_size > mma_qk_tiler_mn[1]:
                print(f"[can_implement] page_size {page_size} > mma_qk_tiler_mn[1] {mma_qk_tiler_mn[1]}.")
                return False
        else:
            if use_page_table and page_size != mma_qk_tiler_mn[1]:
                print(f"[can_implement] use_page_table is True but page_size {page_size} != mma_qk_tiler_mn[1] {mma_qk_tiler_mn[1]}.")
                return False
        if mma_qk_tiler_mn[0] != H or mma_pv_tiler_mn[0] != H:
            print(f"[can_implement] mma_qk_tiler_mn[0] {mma_qk_tiler_mn[0]} or mma_pv_tiler_mn[0] {mma_pv_tiler_mn[0]} != H {H}.")
            return False
        if is_var_split_kv and (not use_page_table or not is_var_seq):
            print(f"[can_implement] is_var_split_kv is True but use_page_table is {use_page_table} or is_var_seq is {is_var_seq} (both must be True).")
            return False
        if is_var_seq and not use_page_table:
            print(f"[can_implement] is_var_seq is True but use_page_table is False.")
            return False
        if H != 128:
            print(f"[can_implement] H {H} != 128.")
            return False
        if K <= 0:
            print(f"[can_implement] K {K} <= 0.")
            return False
        return True

def create_page_table(batch_size, seq_len, is_var_seq, use_page_table, page_size):
    page_table_ref, page_table, page_table_gpu = None, None, None
    if use_page_table:
        max_seq_len = seq_len if not is_var_seq else torch.max(cache_seqs_torch)
        page_count = ceil_div(max_seq_len, page_size)
        page_table_ref = torch.empty([batch_size, page_count], dtype=torch.int32)
        # use transposed index for page table to make sure the value is in bound of `batch_size * seq_len_block`
        for b in range(batch_size):
            for j in range(page_count):
                page_table_ref[b, j] = b + j * batch_size
        page_table_gpu = page_table_ref.permute(1, 0).cuda()
        page_table = from_dlpack(
            page_table_gpu, assumed_align=16
        ).mark_layout_dynamic(leading_dim=0)
    return page_table_ref, page_table, page_table_gpu

def create_block_split_kvs(
    batch_size,
    split_kv,
    cache_seqs_ref,
    is_var_split_kv,
    mma_qk_tiler_mn,
    cluster_shape_mnk,
    max_active_clusters,
):
    block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu = None, None, None
    # check if split_kv is valid otherwise do auto setting of split_kv
    if is_var_split_kv:
        block_split_kvs_ref = torch.zeros([batch_size], dtype=torch.int32)
        for b in range(batch_size):
            block_split_kvs_ref[b] = (
                BlackwellMultiLatentAttentionForward.get_split_kv(
                    batch_size,
                    cache_seqs_ref[b].item(),
                    mma_qk_tiler_mn,
                    max_active_clusters * cluster_shape_mnk[0],
                )
            )
        split_kv = torch.max(block_split_kvs_ref).item()
        block_split_kvs_gpu = block_split_kvs_ref.cuda()
        block_split_kvs = from_dlpack(
            block_split_kvs_gpu, assumed_align=16
        ).mark_layout_dynamic()
    elif split_kv <= 0:
        split_kv = BlackwellMultiLatentAttentionForward.get_split_kv(
            batch_size,
            cache_seqs_ref[0].item(),
            mma_qk_tiler_mn,
            max_active_clusters * cluster_shape_mnk[0],
        )
    return split_kv, block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu

def create_workspace(num_heads, latent_dim, batch_size, split_kv, acc_dtype):
    workspace_size = BlackwellMultiLatentAttentionForward.get_workspace_size(
        num_heads,
        latent_dim,
        batch_size,
        split_kv,
        acc_dtype,
    )

    workspace, workspace_torch = None, None
    if workspace_size > 0:
        workspace_torch = torch.empty([workspace_size], dtype=torch.int8).cuda()
        workspace = from_dlpack(workspace_torch, assumed_align=16)
    return workspace, workspace_torch

class BatchMLAPagedAttentionWrapperCuteDSL:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr

        self._is_persistent = True
        self._is_cpasync = False
        self._use_page_table = True
        self._in_dtype = cutlass.Float16
        self._out_dtype = cutlass.Float16
        self._acc_dtype = cutlass.Float32
        self._lse_dtype = cutlass.Float32
        self._mma_qk_tiler_mn = (128, 128)
        self._mma_pv_tiler_mn = (128, 256)
        self._split_kv = -1


    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan the MLA attention computation.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
            For decoding attention, the length of each query is 1, and the content
            of the tensor should be ``[0, 1, 2, ..., batch_size]``.
        kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.Tensor
            The query length of each request, shape: ``[batch_size]``.
        num_heads : int
            The number of heads in query/output tensor.
        head_dim_ckv : int
            The head dimension of compressed-kv.
        head_dim_kpe : int
            The head dimension for rope k-cache.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to use causal attention.
        sm_scale : float
            The scale factor for softmax operation.
        q_data_type : torch.dtype
            The data type of the query tensor.
        kv_data_type : torch.dtype
            The data type of the kv-cache tensor.
        use_profiler : bool, optional
            Whether to enable intra-kernel profiler, default is False.
        """

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run this example!")

        batch_size = qo_indptr.shape[0] - 1
        seq_len = kv_len_arr.max().item()
        pages_num = math.ceil(seq_len / page_size)

        # check whether kv_len_arr is all the same
        if torch.all(kv_len_arr == kv_len_arr[0]):
            self._is_var_seq = False
            self._is_var_split_kv = False
        else:
            self._is_var_seq = True
            self._is_var_split_kv = False

        self._batch_size = batch_size
        self._seq_len = seq_len
        self._pages_num = math.ceil(seq_len / page_size)
        self._num_heads = num_heads
        self._head_dim_ckv = head_dim_ckv
        self._head_dim_kpe = head_dim_kpe
        self._page_size = page_size
        self._causal = causal
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler

        # use input to create some random tensors
        q_nope = torch.randn(
            batch_size * 1, num_heads, head_dim_ckv, dtype=torch.half, device="cuda"
        )
        q_pe = torch.randn(
            batch_size * 1, num_heads, head_dim_kpe, dtype=torch.half, device="cuda"
        )
        ckv = torch.randn(
            batch_size * pages_num, page_size, head_dim_ckv, dtype=torch.half, device="cuda"
        )
        kpe = torch.randn(
            batch_size * pages_num, page_size, head_dim_kpe, dtype=torch.half, device="cuda"
        )

        if not BlackwellMultiLatentAttentionForward.can_implement(
            batch_size,
            seq_len,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            self._in_dtype,
            self._out_dtype,
            self._acc_dtype,
            self._lse_dtype,
            self._mma_qk_tiler_mn,
            self._mma_pv_tiler_mn,
            self._split_kv,
            self._is_persistent,
            self._is_cpasync,
            self._is_var_seq,
            self._is_var_split_kv,
            self._use_page_table,
            page_size,
        ):
            raise TypeError(
                f"Unsupported testcase {self._in_dtype}, "
                f"{self._out_dtype}, "
                f"{self._acc_dtype}, "
                f"{self._lse_dtype}, {self._mma_qk_tiler_mn}, "
                f"{self._mma_pv_tiler_mn}, {self._split_kv}, {self._is_persistent}, "
                f"{self._is_cpasync}, {self._is_var_seq}, {self._is_var_split_kv}, "
                f"{self._use_page_table}, {page_size}"
            )

        if self._use_cuda_graph:
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=True)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=True)
            self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
            self._kv_len_arr_buf.copy_(kv_len_arr, non_blocking=True)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)

        self._cache_seqs_cute = from_dlpack(kv_len_arr, assumed_align=16)
        self._page_table_ref, self._page_table, self._page_table_gpu = create_page_table(
            batch_size, seq_len, self._is_var_seq, self._use_page_table, page_size
        )
        cluster_shape_mnk = (2, 1, 1)
        hardware_info = utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        )
        self._split_kv, self._block_split_kvs_ref, self._block_split_kvs, self._block_split_kvs_gpu = (
            create_block_split_kvs(
                batch_size,
                self._split_kv,
                kv_len_arr,
                self._is_var_split_kv,
                self._mma_qk_tiler_mn,
                cluster_shape_mnk,
                max_active_clusters,
            )
        )

        q_latent_cute = torch_to_cute(q_nope, self._in_dtype, is_dynamic_layout=True)

        q_rope_cute = torch_to_cute(q_pe, self._in_dtype, is_dynamic_layout=True)

        c_latent_cute = torch_to_cute(ckv, self._in_dtype, is_dynamic_layout=True, page_table=self._page_table, page_size=self._page_size, cache_seqs=kv_len_arr)

        c_rope_cute = torch_to_cute(kpe, self._in_dtype, is_dynamic_layout=True, page_table=self._page_table, page_size=self._page_size, cache_seqs=kv_len_arr)

        self._o_cute, self._o_torch = create_tensor(
            batch_size, num_heads, head_dim_ckv, self._out_dtype, is_dynamic_layout=True
        )
        self._lse_cute, self._lse_torch = create_tensor(
            batch_size, num_heads, 1, self._lse_dtype, is_dynamic_layout=True, is_lse=True,
        )
        self._workspace, self._workspace_torch = create_workspace(
            num_heads, head_dim_ckv, batch_size, self._split_kv, self._acc_dtype
        )

        mla = BlackwellMultiLatentAttentionForward(
            head_dim_ckv,
            head_dim_kpe,
            self._acc_dtype,
            self._lse_dtype,
            self._mma_qk_tiler_mn,
            self._mma_pv_tiler_mn,
            max_active_clusters,
            self._is_persistent,
            self._is_cpasync,
            self._use_page_table,
            self._is_var_seq,
            self._is_var_split_kv,
        )

        # Get current CUDA stream from PyTorch
        torch_stream = torch.cuda.current_stream()
        # Get the raw stream pointer as a CUstream
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # compile mla kernel
        compiled_mla = cute.compile(
            mla,
            q_latent_cute,
            q_rope_cute,
            c_latent_cute,
            c_rope_cute,
            self._page_table,
            self._o_cute,
            self._lse_cute,
            self._workspace,
            self._split_kv,
            self._cache_seqs_cute,
            self._block_split_kvs,
            self._sm_scale,
            1.0,
            stream,
        )

        self._compiled_mla = compiled_mla

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation."""

        if self._compiled_mla is None:
            raise RuntimeError("Plan the MLA attention computation first!")

        assert return_lse is True, "return_lse must be True for CuteDSL implementation"

        q_latent_cute = torch_to_cute(q_nope, self._in_dtype, is_dynamic_layout=True)

        q_rope_cute = torch_to_cute(q_pe, self._in_dtype, is_dynamic_layout=True)

        c_latent_cute = torch_to_cute(ckv_cache, self._in_dtype, is_dynamic_layout=True, page_table=self._page_table, page_size=self._page_size, cache_seqs=kv_len)

        c_rope_cute = torch_to_cute(kpe_cache, self._in_dtype, is_dynamic_layout=True, page_table=self._page_table, page_size=self._page_size, cache_seqs=kv_len)

        # o_cute, o_torch = create_tensor(
        #     self._batch_size, self._num_heads, self._head_dim_ckv, self._out_dtype, is_dynamic_layout=True
        # )
        # lse_cute, lse_torch = create_tensor(
        #     self._batch_size, self._num_heads, 1, self._lse_dtype, is_dynamic_layout=True, is_lse=True,
        # )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._compiled_mla(
            q_latent_cute,
            q_rope_cute,
            c_latent_cute,
            c_rope_cute,
            self._page_table,
            self._o_cute,
            self._lse_cute,
            self._workspace,
            self._split_kv,
            self._cache_seqs_cute,
            self._block_split_kvs,
            self._sm_scale,
            1.0,
            stream,
        )

        return self._o_torch, self._lse_torch


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_to_cute(
        torch_tensor_gpu,
        dtype,
        is_dynamic_layout=True,
        page_table=None,
        page_size=None,
        cache_seqs=None,
    ):
        shape = torch_tensor_gpu.shape
        B, HK, D = shape
        if page_table is not None:
            if cache_seqs is not None:
                max_seq_len = torch.max(cache_seqs)
                shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
            else:
                shape = (B * ceil_div(HK, page_size), page_size, D)

        permute_order = (1, 2, 0)

        # permute torch tensor according to permute_order
        torch_tensor_gpu = torch_tensor_gpu.permute(permute_order)

        stride_order = (2, 0, 1)
        leading_dim = 1

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=leading_dim
            ).mark_compact_shape_dynamic(
                mode=leading_dim,
                stride_order=stride_order,
                divisibility=(128 // dtype.width),
            )

        cute_tensor = cutlass_torch.convert_cute_tensor(
            torch_tensor_gpu,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return cute_tensor

def create_tensor(
    B,
    HK,
    D,
    dtype,
    is_dynamic_layout=True,
    page_table=None,
    cache_seqs=None,
    is_lse=False,
    page_size=None,
):
    shape = (B, HK, D)
    if page_table is not None:
        if cache_seqs is not None:
            max_seq_len = torch.max(cache_seqs)
            shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
        else:
            shape = (B * ceil_div(HK, page_size), page_size, D)
    permute_order = (1, 2, 0)
    stride_order = (2, 0, 1)
    leading_dim = 1
    if is_lse:
        shape = (B, HK)
        permute_order = (1, 0)
        stride_order = (1, 0)
        leading_dim = 0
    init_config = cutlass.torch.RandomInitConfig(min_val=-2, max_val=2)
    torch_dtype = cutlass_torch.dtype(dtype)
    # Create dtype torch tensor (cpu)
    torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        shape,
        torch_dtype,
        permute_order=permute_order,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=init_config,
    )
    # Create dtype torch tensor (gpu)
    torch_tensor_gpu = torch_tensor_cpu.cuda()

    # Create dtype cute tensor (gpu)
    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype

    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        ).mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=(128 // dtype.width),
        )
    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, torch_tensor_gpu

def mla_cutedsl(
    num_heads: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    softmax_scale: float,
    output_scale: float,
    q_nope_torch: torch.Tensor,
    q_pe_torch: torch.Tensor,
    ckv_torch: torch.Tensor,
    kpe_torch: torch.Tensor,
    cache_seqs_torch: torch.Tensor,
):
    """Execute Multi-Latent Attention (MLA) on Blackwell architecture.

    This function performs the complete MLA computation pipeline using the provided input tensors.
    It uses fixed configuration parameters optimized for the Blackwell architecture and returns
    both the attention output and log-sum-exp values.

    :param num_heads: Number of attention heads
    :type num_heads: int
    :param latent_dim: Dimension of query/compressed latent
    :type latent_dim: int
    :param rope_dim: Dimension of query/compressed rope
    :type rope_dim: int
    :param page_size: Page size for the page table
    :type page_size: int
    :param softmax_scale: Attention score scaling factor
    :type softmax_scale: float
    :param output_scale: Output scaling factor
    :type output_scale: float
    :param q_nope_torch: Query latent tensor with shape [batch_size, num_heads, latent_dim]
    :type q_nope_torch: torch.Tensor
    :param q_pe_torch: Query RoPE tensor with shape [batch_size, num_heads, rope_dim]
    :type q_pe_torch: torch.Tensor
    :param ckv_torch: Compressed key-value tensor with shape [batch_size * pages, page_size, latent_dim]
    :type ckv_torch: torch.Tensor
    :param kpe_torch: Key RoPE tensor with shape [batch_size * pages, page_size, rope_dim]
    :type kpe_torch: torch.Tensor
    :param cache_seqs_torch: Cache sequence lengths tensor with shape [batch_size]
    :type cache_seqs_torch: torch.Tensor

    :return: Tuple of (output_tensor, lse_tensor)
    :rtype: tuple[torch.Tensor, torch.Tensor]

    :raises ValueError: If input shapes are incompatible or head dimension is unsupported
    :raises RuntimeError: If GPU is unavailable for computation
    """

    is_persistent = True
    is_cpasync = False
    use_page_table = True
    in_dtype = cutlass.Float16
    out_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    lse_dtype = cutlass.Float32
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)
    split_kv = -1
    batch_size = cache_seqs_torch.shape[0]
    seq_len = torch.max(cache_seqs_torch).item()

    # check whether cache_seqs_torch is all the same
    if torch.all(cache_seqs_torch == cache_seqs_torch[0]):
        is_var_seq = False
        is_var_split_kv = False
    else:
        is_var_seq = True
        is_var_split_kv = False

    print(f"Running Blackwell MLA test with:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  num_heads: {num_heads}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  rope_dim: {rope_dim}")
    print(f"  in_dtype: {in_dtype}")
    print(f"  out_dtype: {out_dtype}")
    print(f"  acc_dtype: {acc_dtype}")
    print(f"  mma_qk_tiler_mn: {mma_qk_tiler_mn}")
    print(f"  mma_pv_tiler_mn: {mma_pv_tiler_mn}")
    print(f"  split_kv: {split_kv}")
    print(f"  is_persistent: {is_persistent}")
    print(f"  is_cpasync: {is_cpasync}")
    print(f"  is_var_seq: {is_var_seq}")
    print(f"  is_var_split_kv: {is_var_split_kv}")
    print(f"  use_page_table: {use_page_table}")
    print(f"  page_size: {page_size}")
    print(f"  softmax_scale: {softmax_scale}")
    print(f"  output_scale: {output_scale}")

    # Prepare pytorch tensors: Q, K, V (random from 0 to 2) and O (all zero)

    torch.manual_seed(1111)


    cache_seqs_cute = from_dlpack(cache_seqs_torch, assumed_align=16)
    page_table_ref, page_table, page_table_gpu = create_page_table(
        batch_size, seq_len, is_var_seq, use_page_table, page_size
    )
    cluster_shape_mnk = (2, 1, 1)
    hardware_info = utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1]
    )
    split_kv, block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu = (
        create_block_split_kvs(
            batch_size,
            split_kv,
            cache_seqs_torch,
            is_var_split_kv,
            mma_qk_tiler_mn,
            cluster_shape_mnk,
            max_active_clusters,
        )
    )


    q_latent_cute = torch_to_cute(q_nope_torch, in_dtype, is_dynamic_layout=True)

    q_rope_cute = torch_to_cute(q_pe_torch, in_dtype, is_dynamic_layout=True)

    c_latent_cute = torch_to_cute(ckv_torch, in_dtype, is_dynamic_layout=True, page_table=page_table, page_size=page_size, cache_seqs=cache_seqs_torch)

    c_rope_cute = torch_to_cute(kpe_torch, in_dtype, is_dynamic_layout=True, page_table=page_table, page_size=page_size, cache_seqs=cache_seqs_torch)

    o_cute, o_torch = create_tensor(
        batch_size, num_heads, latent_dim, out_dtype, is_dynamic_layout=True
    )
    lse_cute, lse_torch = create_tensor(
        batch_size, num_heads, 1, lse_dtype, is_dynamic_layout=True, is_lse=True
    )
    workspace, workspace_torch = create_workspace(
        num_heads, latent_dim, batch_size, split_kv, acc_dtype
    )

    mla = BlackwellMultiLatentAttentionForward(
        latent_dim,
        rope_dim,
        acc_dtype,
        lse_dtype,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        max_active_clusters,
        is_persistent,
        is_cpasync,
        use_page_table,
        is_var_seq,
        is_var_split_kv,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # compile mla kernel
    compiled_mla = cute.compile(
        mla,
        q_latent_cute,
        q_rope_cute,
        c_latent_cute,
        c_rope_cute,
        page_table,
        o_cute,
        lse_cute,
        workspace,
        split_kv,
        cache_seqs_cute,
        block_split_kvs,
        softmax_scale,
        output_scale,
        stream,
    )

    compiled_mla(
            q_latent_cute,
            q_rope_cute,
            c_latent_cute,
            c_rope_cute,
            page_table,
            o_cute,
            lse_cute,
            workspace,
            split_kv,
            cache_seqs_cute,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
    )

    return o_torch, lse_torch


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    def parse_mma_tiler(s: str) -> Tuple[int, int, Tuple[int, int]]:
        ret = parse_comma_separated_ints(s)
        if len(ret) != 4:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected 4 comma-separated integers."
            )
        return (ret[0], ret[1], (ret[2], ret[3]))

    parser = argparse.ArgumentParser(description="Example of MLA on Blackwell.")

    parser.add_argument(
        "--in_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Input data type",
    )

    parser.add_argument(
        "--out_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Output data type",
    )

    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        help="Accumulator data type",
    )

    parser.add_argument(
        "--lse_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        help="LSE data type",
    )
    parser.add_argument(
        "--mma_qk_tiler_mn",
        type=parse_mma_tiler,
        default=(128, 128),
        help="MMA tile shape (H, K)",
    )
    parser.add_argument(
        "--mma_pv_tiler_mn",
        type=parse_mma_tiler,
        default=(128, 256),
        help="MMA tile shape (H, D)",
    )

    parser.add_argument(
        "--is_persistent",
        action="store_true",
        help="Is persistent",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length of K/V",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=128,
        help="Number of heads of Q",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=512,
        help="Latent dimension of Q/C",
    )

    parser.add_argument(
        "--rope_dim",
        type=int,
        default=64,
        help="Rope dimension of Q/C",
    )

    parser.add_argument(
        "--is_cpasync",
        action="store_true",
        help="Use cpasync for load or not",
    )

    parser.add_argument(
        "--is_var_seq",
        action="store_true",
        help="Use variable length of sequence length or not",
    )

    parser.add_argument(
        "--is_var_split_kv",
        action="store_true",
        help="Use variable length of split kv or not",
    )

    parser.add_argument(
        "--use_page_table",
        action="store_true",
        default=True,
        help="Use page table or not, must be True when is_cpasync is True",
    )

    parser.add_argument(
        "--page_size",
        type=int,
        default=128,
        help="Page size of page table",
    )

    parser.add_argument(
        "--split_kv",
        type=int,
        default=-1,
        help="Split KV setting",
    )

    parser.add_argument(
        "--softmax_scale",
        type=float,
        default=1.0,
        help="Scaling factor to scale softmax",
    )

    parser.add_argument(
        "--output_scale",
        type=float,
        default=1.0,
        help="Scaling factor to scale output",
    )

    parser.add_argument(
        "--tolerance", type=float, default=1e-02, help="Tolerance for validation"
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        help="Use cold L2 cache",
    )

    args = parser.parse_args()

    q_nope_torch = torch.randn(
        args.batch_size * 1, args.num_heads, args.latent_dim, dtype=torch.half, device="cuda"
    )
    q_pe_torch = torch.randn(
        args.batch_size * 1, args.num_heads, args.rope_dim, dtype=torch.half, device="cuda"
    )

    pages_num = math.ceil(args.seq_len / args.page_size)
    ckv_torch = torch.randn(
        args.batch_size * pages_num, args.page_size, args.latent_dim, dtype=torch.half, device="cuda"
    )
    kpe_torch = torch.randn(
        args.batch_size * pages_num, args.page_size, args.rope_dim, dtype=torch.half, device="cuda"
    )

    # Create cache_seqs_torch tensor for variable sequence length
    cache_seqs_torch = torch.full([args.batch_size], args.seq_len, dtype=torch.int32, device="cuda")

    output_torch, lse_torch = mla_cutedsl(
        args.num_heads,
        args.latent_dim,
        args.rope_dim,
        args.page_size,
        args.softmax_scale,
        args.output_scale,
        q_nope_torch,
        q_pe_torch,
        ckv_torch,
        kpe_torch,
        cache_seqs_torch,
    )

    print(output_torch.shape)
    print(output_torch)

    print(lse_torch.shape)
    print(lse_torch)
