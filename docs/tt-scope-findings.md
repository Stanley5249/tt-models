# tt-scope findings from optimizing Gemma 4 decode

Context: tt-scope v0.4.0 (`e9f5941`, pinned in this repo's `pixi.toml`), ttnn
0.81.1+conda.8, N300 (1x2 mesh), Ubuntu 22.04. Nine `tracy` captures of a traced
batch-1 decode step (`scripts/profile_workload.py --trace`), 3 steps each,
2,199 down to 574 ops per chip. Ranked by how much each item cost me.

## 1. Op-to-op latency is meaningless for trace replay (high)

- **Trying to do:** find how much of a traced decode step is device idle time
  between ops. The brief assumed ~19 ms of gaps in a 47 ms step.
- **Reproduce:** capture any workload that replays a metal trace, then read
  `OP TO OP LATENCY [ns]` in `ops_perf_results_*.csv`, or `scripts/op_summary.py`.
  The decode step summed to 395 ms of "gaps" for a 34 ms step.
- **Impact:** the column measures host time between trace replays, so it hides the
  real answer: 0.8 ms of gaps per 34 ms step. Without recomputing gaps I would have
  chased dispatch overhead that doesn't exist under trace.
- **Workaround:** `scripts/trace_breakdown.py` recomputes gaps from
  `DEVICE FW START/END CYCLE`, per chip and phase.
- **Fix:** when ops carry a `METAL TRACE REPLAY SESSION ID`, compute op-to-op gaps
  from device firmware cycles inside the replay, and label the column as such.

## 2. No decode-step or per-layer breakdown (high)

- **Trying to do:** rank costs inside one decode step by op position in a layer.
- **Reproduce:** any multi-layer model capture; the CSV is a flat op list.
- **Impact:** every round of the loop needed a custom fold (`--layer-ops N`) that
  breaks whenever an optimization changes the op count (73 -> 57 -> 30 -> 19 -> 18
  ops per layer), and a 31st "layer" silently absorbed the LM head once.
- **Fix:** a `tt-scope summary <run> --phase decode1 --period N` (or auto-detect the
  repeating op-code sequence) that prints per-position mean/max kernel time, cores
  and gap, plus the non-periodic head and tail, per chip.

## 3. Generic ops are anonymous (high for custom-kernel work)

- **Trying to do:** tell my nine `ttnn.generic_op` kernels apart in the profile.
- **Reproduce:** profile any `ttnn.generic_op` program; every row is
  `GenericOpDeviceOperation`.
- **Impact:** I had to identify kernels by input widths and core count (for
  example `768[768] c44` is the expert gate|up). Two ops with the same shapes
  would be indistinguishable.
- **Fix:** take the op name from the first kernel source basename (or a user label
  on the ProgramDescriptor) in the CSV and every summary.

## 4. Per-RISC durations exist but are never summarized (medium)

- **Trying to do:** find why my fused tail ops were slower than their math.
- **Reproduce:** the CSV has `DEVICE BRISC/NCRISC/TRISC0-2 KERNEL DURATION [ns]`;
  no tt-scope view shows them.
- **Impact:** that view showed the readers (NCRISC) took 12-16 us before compute
  could start in three of my kernels, which led to a fix. I found it only by
  reading CSV headers.
- **Fix:** add per-RISC columns to the summary, and flag ops where one RISC's
  duration is close to the op's (the bottleneck RISC).

## 5. Raw logs are huge and duplicated (medium)

- **Reproduce:** a 3-step decode capture: `.logs/profile_log_device.csv` (1.7 GB),
  the same file again under `reports/` (1.7 GB), and `.logs/tracy_ops_times.csv`
  (1.7 GB). The full-model base capture was 11 GB, all runs 63 GB.
- **Impact:** disk use, and ~2 min of post-processing per capture. Only the 28 MB
  ops CSV and the 90 MB `.tracy` were needed afterwards; I deleted the rest.
- **Fix:** keep one copy, compress the device log, or add `--keep-raw no` that drops
  the raw logs after the reports are built.

## 6. Overflow check can't say which phase overflowed (medium)

- **Trying to do:** capture warmup (prefill, compile, trace capture) plus three
  measured steps with `--programs 6000`.
- **Reproduce:** run a workload whose warmup exceeds the buffer but whose measured
  phases don't, with `ReadDeviceProfiler` between phases.
- **Impact:** `check` failed with "the device buffer overflowed"; I had to compare
  `run.log` timestamps with signpost times to see the overflow was in the warmup
  and the three measured steps were intact (2,199 ops each).
- **Fix:** report overflows per signpost phase (or per `ReadDeviceProfiler` read),
  and print the measured programs per RISC so `--programs` can be set from data.

## 7. Summary lines bury the failure (low)

- **Reproduce:** the first capture printed `fail marker warnings` above eight `ok`
  lines, then "the check failed"; grepping for the summary showed only the `ok`
  lines at first.
- **Fix:** print failures last or repeat them next to "the check failed".

## 8. Running from ~/repo/tt-scope doesn't work for a project with its own deps (low)

- **Trying to do:** follow the brief and run `pixi run tt-scope ...` from
  `~/repo/tt-scope`.
- **Impact:** that environment lacks the model's dependencies (transformers, this
  package). I used the tt-scope pinned in tt-vllm's own environment instead.
- **Fix:** document "install tt-scope into the workload's environment" as the
  supported path, or let `tt-scope tracy --python <env python>` pick the target env.

## 9. Per-chip differences are easy to miss (low)

- **Observation:** chip 1 is consistently 0.5 ms slower per step than chip 0
  (15.53 vs 15.01 ms kernel time), likely harvesting and DRAM-bank worker placement
  that my stream kernels compute from chip 0. The flat summaries report chips
  separately but never compare them.
- **Fix:** a per-chip diff of the per-position table.

## 10. Leftover device state from killed processes isn't flagged (low)

- **Observation:** after a killed vLLM server, the next run logs
  `TT_FATAL: Read unexpected run_mailbox value` at device init, then recovers.
  `check` doesn't mention it; a reader of `run.log` may think the capture is bad.
- **Fix:** classify init-time "dispatch cores still running" messages as a warning
  in `check`, with a note that they come from a previous process.
