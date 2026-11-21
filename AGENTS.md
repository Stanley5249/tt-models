# tt-models

## Checks

- `just check` runs the format checks, ruff and pyrefly.
- `just test` runs the host tests, and `just test -m device` the N300 ones.

## Common mistakes

Seen on this machine; the root causes are unknown.

- After a server stops, the next device job can fail at init with "Timeout waiting for Ethernet core service remote IO request". Run `tt-smi -r` and log the reset in `logs/card-resets.log`.
- A host-only ttnn conversion, such as `ttnn.from_torch(..., dtype=ttnn.bfloat4_b)` without a device, starts the device cluster and can break a device job running next to it. CPU-only code emulates BFP formats in torch instead, as `scripts/accuracy/q4_emulation.py` does.
- Killing a wrapper's PID, such as `pixi run bash script.sh`, can leave the server running on its port. Stop the PID that listens on the port, then check that the port is free.
