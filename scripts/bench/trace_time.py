"""Blocking trace execution, readback and host softcap of a batch-1 decode step
(diagnostic).
"""

import statistics
import time

import torch
import ttnn

from tt_models.common.mesh import open_n300
from tt_models.models.gemma4.checkpoint import Checkpoint
from tt_models.models.gemma4.model import Gemma4TT

with open_n300(trace_region_size=300_000_000) as mesh:
    m = Gemma4TT(mesh, Checkpoint(), max_batch=1)
    tables = m.allocate_kv_cache(1, 128)
    m.prefill(torch.randint(100, 2000, (64,)), 0, tables[:, 0])
    m.decode(torch.tensor([5]), torch.tensor([64]), tables, trace=True)
    tid, dev, out = m.traces[1]
    print("out", out.shape, out.layout, out.memory_config().memory_layout)
    host = m._decode_inputs(torch.tensor([5]), torch.tensor([70]), tables)
    r: list[float] = []
    c: list[float] = []
    e: list[float] = []
    full: list[float] = []
    sc: list[float] = []
    am: list[float] = []
    sc1: list[float] = []
    for _ in range(20):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for k, v in host.items():
            ttnn.copy_host_to_device_tensor(v, dev[k])
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t2 = time.perf_counter()
        x = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        t3 = time.perf_counter()
        c.append(t1 - t0)
        e.append(t2 - t1)
        r.append(t3 - t2)
        t0 = time.perf_counter()
        for k, v in host.items():
            ttnn.copy_host_to_device_tensor(v, dev[k])
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        x = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        full.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        y = x.float().reshape(-1, m.vocab).div_(30).tanh_().mul_(30)
        sc.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        a = y.argmax(-1)
        am.append(time.perf_counter() - t0)
        torch.set_num_threads(1)
        t0 = time.perf_counter()
        y = x.float().reshape(-1, m.vocab).div_(30).tanh_().mul_(30)
        sc1.append(time.perf_counter() - t0)
        torch.set_num_threads(20)

    def f(v: list[float]):
        return round(statistics.median(v) * 1e3, 3)

    print(
        "copies+sync",
        f(c),
        "execute",
        f(e),
        "read",
        f(r),
        "pipelined copies+execute+read",
        f(full),
        "softcap",
        f(sc),
        "argmax",
        f(am),
        "softcap 1 thread",
        f(sc1),
    )
    m.release_traces()
