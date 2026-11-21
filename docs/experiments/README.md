# Experiments

One file per decision, named `YYYY-MM-DD-short-topic.md` after the day the
decision was made. Data the doc cites sits in the folder of the same name.
Numbers live here and nowhere else: the README and the guides link to these
files instead of repeating them, so a new measurement only touches a new doc.

Template:

    # Title

    **Question.** What was unknown, and why it mattered.

    **Setup.** Commit(s), machine state, and the exact command.

    **Result.** The numbers, with run-to-run noise where it matters.

    **Decision.** Kept, rejected or deferred, and why.

    **Data.** `<same-name folder>/` or none.

A doc is a record of its day. When a later change makes one obsolete, add a
new doc that links back rather than editing the old numbers. Scripts that
only served one experiment are removed from `scripts/`; the doc names the
commit where they last existed (`git show <commit>:<path>`). Reports kept in a
data folder cite script and result paths as they were at their commit.

The measurement rules every doc follows:

- Decode accuracy is teacher-forced against the fp32 HF reference of the same
  checkpoint. One run moves 0.04-0.07 KL under a one-ulp input change, so a
  comparison uses 3 seeds or more (seed 0 plain, others with 2^-9 embedding
  noise) and the mean ([noise floor](2026-11-17-accuracy-noise-floor.md)).
- Timing changes are judged against a same-day baseline, with the CPU
  governor at `performance`.
