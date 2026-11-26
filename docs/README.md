# Documentation

- [Gemma 4 optimization using tt-scope](reports/gemma4-optimization-with-tt-scope.md): how the agent used profiling evidence to choose and validate optimizations.
- [Supporting results and limits](reports/gemma4-optimization-with-tt-scope-appendix.md): additional changes, serving behavior, measurements, and validation limits.
- [Glossary](glossary.md): reusable terminology maintained within tt-models.
- [Serving](serving.md): backend behavior, supported features, and validation commands.
- [Experiments](experiments/README.md): dated decision records, measurement rules, and their original data.

## Writing reports

Reports in `reports/` summarize work across experiments. Raw data and dated decisions stay in `experiments/`.

### Content

- Keep the optimizations in the main report, both major and minor. Move tool details, serving stability fixes, rejected attempts, validation limits, and each item's measurement conditions to the appendix.
- Write the main report for a human reader. Show only what helps the reader follow the story: the cause, the change, and the result.
- Write each major optimization as three parts: 觀察, 做法, and 結果. Write each minor one as one line with the same three parts.
- In 觀察, name the code path that existed before the change, such as a default `ttnn` function, and the data source, such as a Tracy capture, `py-spy`, a code review, or a kernel test. Give the cause and the one number that shows it.
- In 做法, explain how the change makes the work faster, without a deep technical walkthrough. Link the main source code.
- In 結果, give one headline number before and after the change, with its timer scope and batch size. Prefer timing without the profiler, such as decode latency timed by `model.decode` or prefill time, over Tracy kernel time. Add one serving number, such as TPOT, TTFT, or output throughput, when one exists.
- Order the optimizations so that they read as one story, and leave out what does not belong to it.

### Evidence

- Give every claim and every number its own source link. Don't merge the sources of separate items into one shared list.
- Use "baseline" only for the real starting point. For other comparisons, write 改動前 and 改動後.
- Don't add up gains measured under different runtimes or settings. When a comparison includes several changes, say which ones.
- Keep a limit next to the result when it changes how to read the number, such as a timer that excludes token selection or a path that only runs for greedy decoding.
- Put the rest in the appendix's measurement conditions for that item: experiment dates, commits, other measurements, and other changes that the same comparison includes.
- Don't write commit hashes or comparison ranges in the main report. In the appendix, link a commit from words, such as [Commit] or [LM head commit].

### Terms and style

- Use the terms in [the glossary](glossary.md), and add a reusable term there before using it. A report may define a term briefly on first use.
- In the glossary, give each term its own row, and keep related terms in adjacent rows. Don't join several terms with ／ in one row, because a reader then has to match each term to its position in the definition.
- Don't invent names for techniques. Describe what the change does.
- Refer to an experiment by the date in its record's file name, such as 2026-11-17, and link the record. Put dates in the appendix, not in the main report's items. Don't use round, stage, early, or late, because the experiment records, the slides, and the agent prompts give these words different meanings.
- Keep formal technical terms in American English.
- Write device and host in English, not as 裝置 or 主機, so that each has one name.
- Write row only for a row of a tile or tensor, and batch size for the number of requests in a step. Don't use 列.
- Write the time of one decode step as decode latency, and state the timer before it in plain words, such as 以 `model.decode` 計時的 decode latency. Don't put the timer in parentheses. Write step as decode step or scheduler step. Don't use 單步, 每步 or 一步.
- Link files that should stay current, such as source files, scripts, and experiment records, with relative paths. Link a past state, such as a commit, a comparison, or a specific line in source code, with a GitHub URL pinned to a commit, because line numbers move.
- Cut filler, recaps, notes about how the report was written, and repeated cross references. Keep every number, condition, and limit in the main report or the appendix.
- Keep sentences complete when you shorten them. Name the subject and the object, such as which component reads what data from where.
