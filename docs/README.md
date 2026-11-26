# Documentation

- [Gemma 4 optimization using tt-scope](reports/gemma4-optimization-with-tt-scope.md): how the agent used profiling evidence to choose and validate optimizations.
- [Supporting results and limits](reports/gemma4-optimization-with-tt-scope-appendix.md): additional changes, serving behavior, measurements, and validation limits.
- [Glossary](glossary.md): reusable terminology maintained within tt-models.
- [Serving](serving.md): backend behavior, supported features, and validation commands.
- [Experiments](experiments/README.md): dated decision records, measurement rules, and their original data.

## Writing reports

Reports in `reports/` summarize work across experiments. Raw data and dated decisions stay in `experiments/`.

### Content

- Keep the optimizations in the main report, both major and minor. Move tool details, serving stability fixes, rejected attempts, and validation limits to the appendix.
- Write each major optimization as three parts: 觀察, 做法, and 結果. Write each minor one as one line with the same three parts.
- In 觀察, name the code path that existed before the change, such as a default `ttnn` function, and the data source, such as a Tracy capture, `py-spy`, a code review, or a kernel test on synthetic input.
- In 做法, explain how the change makes the work faster, without a deep technical walkthrough. Link the main source code.
- In 結果, give the numbers before and after the change, with the timer scope, batch size, and the commit or runtime version that was measured.
- Order the optimizations so that they read as one story, and leave out what does not belong to it.

### Evidence

- Give every claim and every number its own source link. Don't merge the sources of separate items into one shared list.
- Use "baseline" only for the real starting point. For other comparisons, write 改動前 and 改動後.
- Don't add up gains measured under different runtimes or settings. When a comparison includes several changes, say which ones.
- Keep limits next to the result they limit, such as a timer that excludes token selection.

### Terms and style

- Use the terms in [the glossary](glossary.md), and add a reusable term there before using it. A report may define a term briefly on first use.
- Don't invent names for techniques. Describe what the change does.
- Keep formal technical terms in American English.
- Link files that should stay current, such as source files, scripts, and experiment records, with relative paths. Link a past state, such as a commit, a comparison, or a specific line in source code, with a GitHub URL pinned to a commit, because line numbers move.
- Cut filler, recaps, notes about how the report was written, and repeated cross references. Keep every number, condition, and limit.
