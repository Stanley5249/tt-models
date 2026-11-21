"""Q4_0 factoring of QAT weights, torch only (no ttnn, so host emulation can use it).

The QAT checkpoint stores every 32-value block as q * s: integers q in -8..7 and one
learned step s,
rounded to BF16.
"""

import torch

BLOCK = 32
STEP_TOL = 0.05  # |w / s - round(w / s)| of a BF16-rounded q * s stays below 2^-9 * 8


def factor_blocks(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor each weight block into integers and a shared step.

    B [..., 32] -> integers q (-8..8, unclamped) and steps s [..., 1] with b ~= q *
    s.

    The step of a block is its smallest nonzero |b| divided by the smallest k in 1..8
    that makes
    every b / s an integer of at most 8. A block of zeros gets q = 0 and s = 0. Raises
    if a block
    has no such step, since every caller would then misstate it.
    """
    nz = torch.where(b != 0, b.abs(), torch.full_like(b, float("inf")))
    s0 = nz.amin(-1, keepdim=True)
    del nz
    zero = torch.isinf(s0)
    s0 = torch.where(zero, torch.ones_like(s0), s0)
    s = s0.clone()
    done = zero.clone()
    for k in range(1, 9):
        r = b / (s0 / k)
        ok = ((r - r.round()).abs().amax(-1, keepdim=True) < STEP_TOL) & (
            r.round().abs().amax(-1, keepdim=True) <= 8
        )
        s = torch.where(ok & ~done, s0 / k, s)
        done |= ok
    if not bool(done.all()):
        raise ValueError(
            f"{int((~done).sum())} blocks are not integer multiples of one step within "
            "-8..8"
        )
    q = (b / s).round()
    return q, torch.where(zero, torch.zeros_like(s), s)
