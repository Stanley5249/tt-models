"""Device time of an op, measured by trace replay."""

import time

import ttnn


def device_time_us(mesh: ttnn.MeshDevice, fn, iters: int = 50) -> float:
    """Measure device time per call.

    Device time per call: replay a trace of `iters` calls, so host dispatch is
    excluded.
    """
    fn().deallocate(True)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        fn().deallocate(True)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / iters * 1e6
    ttnn.release_trace(mesh, tid)
    return round(dt, 1)
