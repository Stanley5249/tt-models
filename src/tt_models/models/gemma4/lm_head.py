"""The tied LM head: the embedding table as a DRAM-sharded matmul split over the vocabulary."""

import numpy as np
import torch
import ttnn

from tt_models.common.config import DRAM, HIFI2, TILE, dram_sharded, width_sharded_l1
from tt_models.common.norm import rms_norm
from tt_models.common.weights import norm_weight

# The LM head is read as LM_SPLITS DRAM-sharded matmuls per chip: 2.5 -> 1.37 ms per
# decode step against one interleaved matmul (scripts/bench_lm_head_dram.py).
# A single DRAM-sharded matmul over all 65536 columns does not fit L1.
LM_SPLITS = 16
LM_CORES = 8


class LmHead:
    def __init__(self, mesh, loader, ckpt, cfg, dtype):
        self.mesh = mesh
        self.cfg = cfg
        self.tp = mesh.get_num_devices()
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.final_norm = loader.put("final_norm", lambda: norm_weight(ckpt.get("norm.weight")), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.weights = self._weights(loader, ckpt, dtype)
        self.lm_in = width_sharded_l1(self.hidden, LM_CORES)

    def _weights(self, loader, ckpt, dtype) -> list[ttnn.Tensor]:
        """LM_SPLITS DRAM-sharded column blocks per chip; chip c split i holds vocab
        columns c * V/tp + i * width onwards, so concatenating the splits on each chip
        and then the chips gives the vocabulary order."""
        per_chip = self.vocab // self.tp
        width = per_chip // LM_SPLITS
        mem = dram_sharded(self.mesh, self.hidden, width)
        self.lm_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.hidden // TILE // LM_CORES, per_core_M=1, per_core_N=width // TILE // LM_CORES, fused_activation=None
        )
        self.lm_out = width_sharded_l1(width, LM_CORES)
        table = {}

        def split(i):
            if "t" not in table:
                table["t"] = ckpt.get("embed_tokens.weight").T  # [H, V]
            t = table["t"]
            cols = [t[:, c * per_chip + i * width : c * per_chip + (i + 1) * width] for c in range(self.tp)]
            return torch.cat(cols, -1).contiguous()[None, None]

        heads = [loader.put(f"lm_head_ds{LM_SPLITS}_{i}", lambda i=i: split(i), dtype, shard_dim=-1, memory_config=mem) for i in range(LM_SPLITS)]
        table.clear()
        return heads


    def logits(self, x: ttnn.Tensor) -> torch.Tensor:
        rows = x.shape[2]
        if rows > TILE:
            # ttnn.linear segfaults on the host for M > 32 against the 131072-wide
            # LM head shard (ttnn 0.81), so multi-row logits go one tile at a time.
            parts = []
            for start in range(0, rows, TILE):
                part = ttnn.slice(x, [0, 0, start, 0], [1, 1, min(start + TILE, rows), self.hidden])
                parts.append(self.logits(part))  # row slices are views of x; do not free them
            return torch.cat(parts)
        out = self.device(x)
        logits = self.read(out)
        out.deallocate(True)
        return logits

    def device(self, x: ttnn.Tensor, normed: bool = False) -> ttnn.Tensor:
        """Raw fp32 logits, row-major; the host applies the final softcap
        (three fp32 ops over the tile-padded [32, 65536] took 0.63 ms here)."""
        h = x if normed else rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        hs = ttnn.to_memory_config(h, self.lm_in)
        if not normed:
            h.deallocate(True)
        rows = x.shape[2]
        parts = []
        for w in self.weights:
            o = ttnn.linear(hs, w, program_config=self.lm_cfg, compute_kernel_config=HIFI2, memory_config=self.lm_out, dtype=ttnn.float32)
            # Untilizing straight from L1 drops the tile padding, so the host reads
            # B rows instead of 32 (9 ms -> under 1 ms for 131072 fp32 logits), and
            # the row-major concat below only moves those rows.
            parts.append(ttnn.untilize_with_unpadding(o, output_tensor_end=[0, 0, rows - 1, o.shape[3] - 1], memory_config=DRAM))
            o.deallocate(True)
        hs.deallocate(True)
        out = ttnn.concat(parts, dim=-1)
        for o in parts:
            o.deallocate(True)
        return out

    def read(self, out: ttnn.Tensor, softcap: bool = True) -> torch.Tensor:
        """fp32 logits [rows, V] of a row-major [1, 1, rows, V/tp] per-chip output.
        (Copying into a kept host tensor and converting its shards was slower:
        1.0 ms against 0.66 ms for one row.)"""
        logits = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float().reshape(-1, self.vocab)
        return self.softcap(logits) if softcap else logits

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the final logit softcap in place (cap * tanh(x / cap)) and return logits.

        It costs 0.34 ms for one row of 262,144 logits and 1.35 ms for four, and it
        keeps the order, so greedy and top-k sampling can cap only their candidates.
        numpy ufuncs run on this thread: torch's pool sleeps during the device wait,
        and waking it once made the same softcap take 1 ms instead of 0.2 ms."""
        cap = self.cfg.final_logit_softcapping
        if cap:
            arr = logits.numpy()
            np.divide(arr, cap, out=arr)
            np.tanh(arr, out=arr)
            np.multiply(arr, cap, out=arr)
        return logits
