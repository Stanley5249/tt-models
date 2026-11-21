# tt-models agent notes

## Observed behaviour (unverified, dated)

These were seen on this machine (N300d, ttnn 0.81.1+conda.8, vLLM 0.32.0)
and have no confirmed root cause. Update or remove an entry when it is
confirmed or no longer reproduces.

- **Ethernet timeout after a server stop** (2026-11-21, 3 times). After
  `serve-gemma4` stopped, the next device job failed at init with "Timeout
  waiting for Ethernet core service remote IO request". `tt-smi -r` cleared it
  each time. Each reset is logged in `logs/card-resets.log`.
- **Host-only ttnn calls start the device cluster** (2026-11-21, once). A
  CPU-only script that called `ttnn.from_torch(..., dtype=bfloat4_b)` without a
  device logged "Starting devices in cluster". Run next to a device job, it
  coincided with an Ethernet flush timeout and a reset. The CPU emulation in
  `scripts/accuracy/q4_emulation.py` uses a torch BFP4 emulator instead.
- **A `pixi run bash` wrapper does not pass `kill` on** (2026-11-21, once). A
  server started as `pixi run bash script.sh` kept running after `kill` of the
  recorded PID. The next server could not bind port 8000, while a health
  check still answered from the old one.
