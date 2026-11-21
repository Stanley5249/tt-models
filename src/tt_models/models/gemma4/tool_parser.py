"""Gemma 4 tool parser that enforces required and named tool choice.

    vllm serve ... --tool-parser-plugin \\
        src/tt_models/models/gemma4/tool_parser.py --tool-call-parser gemma4_tt

vLLM 0.32's `gemma4` parser sets no structural tag, so `tool_choice="required"`
or a named tool only reaches the model as prompt text (vllm-project/vllm#50477,
#53363). vLLM's chat path already turns a parser's `structural_tag_model` into
an xgrammar structural tag: the grammar forces the native
`<|tool_call>call:name{...}<tool_call|>` envelope, at least one call for
"required" and exactly the named function for a named choice. xgrammar 0.2.9
ships that tag for Gemma 4 (`gemma_4`, with its `<|"|>` strings), but vLLM 0.32
only knows the xgrammar tags of 0.2.8, so this file registers it as a
vLLM-side builder. Parsing stays the stock gemma4 parser, which reads the
native syntax for every tool choice.

Tool arguments follow the tool's schema only for `strict` tools, as vLLM does
for every model (`--tool-strict-level parameter` enforces all of them). A
non-strict tool still gets an object for its arguments: xgrammar would accept no
argument block at all (`call:name<tool_call|>`), which the gemma4 parser cannot
read as a call.
"""

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.structural_tag_registry import register_vllm_structural_tag
from xgrammar.builtin_structural_tag import get_gemma_4_structural_tag

TAG_MODEL = "gemma_4"


def _object_arguments(tool):
    """A non-strict tool as a strict one whose arguments are any object."""
    if tool.function.strict is not False:
        return tool
    function = tool.function.model_copy(
        update={"strict": True, "parameters": {"type": "object"}}
    )
    return tool.model_copy(update={"function": function})


@register_vllm_structural_tag(TAG_MODEL)
def gemma4_structural_tag(
    function_tools, builtin_tools, tool_choice, reasoning, token_suffix
):
    """vLLM's builder signature over xgrammar's Gemma 4 tag (no token suffix in this
    format).
    """
    if token_suffix:
        raise ValueError(
            f"the Gemma 4 structural tag takes no token suffix, got {token_suffix!r}"
        )
    return get_gemma_4_structural_tag(
        tools=[_object_arguments(t) for t in function_tools],
        builtin_tools=builtin_tools,
        tool_choice=tool_choice,
        reasoning="enabled" if reasoning else "disabled",
    )


@ToolParserManager.register_module("gemma4_tt")
class Gemma4TTToolParser(Gemma4EngineToolParser):
    structural_tag_model = TAG_MODEL
