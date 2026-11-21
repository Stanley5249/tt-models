# Required and named tool choice

**Question.** `tool_choice="required"` or a named tool reached the model only
as prompt text, so a greeting with `required` got a prose answer
(vllm-project/vllm#50477, #53363). Can it be enforced without patching vLLM?

**Setup.** `52aee0d`, `d7145b4`. vLLM 0.32's chat path turns a parser's
`structural_tag_model` into an xgrammar structural tag. xgrammar 0.2.9 added
the `gemma_4` tag (the native `<|tool_call>call:name{...}<tool_call|>`
envelope and Gemma's `<|"|>` strings). Checked with
`scripts/serve/serve_check.py tools`: `auto` with and without a needed tool,
`required` for a greeting, and a named tool for a question that fits the
other tool, with thinking on and off.

**Result.** With the tag, `required` always produces a call and a named
choice exactly that function; the streamed `required` call parses too.
Parsing is the stock gemma4 parser, so `auto` is unchanged. A non-strict
tool's arguments must still be an object, because xgrammar would otherwise
accept a call with no argument block, which the parser cannot read (found in
review, fixed in `d7145b4`).

**Decision.** Kept in this repository: `pixi.toml` overrides vLLM's
`xgrammar==0.2.8` pin with 0.2.9, and
`src/tt_models/models/gemma4/tool_parser.py` registers the tag and the
`gemma4_tt` parser. Drop both once vLLM sets `structural_tag_model` for
gemma4 and pins xgrammar >= 0.2.9.

**Data.** None beyond the commits.
